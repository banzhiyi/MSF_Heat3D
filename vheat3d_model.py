import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class LayerNorm3d(nn.Module):
    """3D LayerNorm that always returns channels-first ordering: (B, C, S, H, W).
    It accepts inputs in either (B, C, S, H, W) or (B, H, W, S, C).
    """

    def __init__(self, hidden_dim, eps=1e-6, debug=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.hidden_dim = hidden_dim
        self.debug = debug

    def forward(self, x: torch.Tensor):
        # Expect 5D input
        if x.dim() != 5:
            raise ValueError(f"LayerNorm3d expects 5D input, got {x.dim()}D")

        # Case A: channels-first (B, C, S, H, W)
        if x.shape[1] == self.hidden_dim:
            B, C, S, H, W = x.shape
            # bring channels to last for LayerNorm: (B*S*H*W, C)
            x_reshaped = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
            x_norm = self.norm(x_reshaped)
            # restore to channels-first (B, C, S, H, W)
            x_norm = x_norm.reshape(B, S, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
            if self.debug:
                print(f"LayerNorm3d: input channels-first {x.shape} -> output {x_norm.shape}")
            return x_norm

        # Case B: channels-last (B, H, W, S, C)
        elif x.shape[-1] == self.hidden_dim:
            B, H, W, S, C = x.shape
            # reshape to (B*S*H*W, C)
            x_reshaped = x.reshape(-1, C)
            x_norm = self.norm(x_reshaped)
            # IMPORTANT: return channels-first (B, C, S, H, W)
            x_norm = x_norm.reshape(B, H, W, S, C).permute(0, 4, 3, 1, 2).contiguous()
            if self.debug:
                print(f"LayerNorm3d: input channels-last {x.shape} -> output {x_norm.shape}")
            return x_norm

        else:
            raise ValueError(f"LayerNorm3d cannot determine channel dimension. "
                             f"Expect hidden_dim={self.hidden_dim} at dim 1 or dim -1, got shape {x.shape}")



class to_channels_first_3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 4, 1, 2, 3).contiguous()


class to_channels_last_3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 4, 1).contiguous()


class StemLayer3D(nn.Module):
    """3D Stem layer that preserves spatial and spectral resolution"""

    def __init__(self,
                 in_chans=1,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        # 使用3D卷积处理光谱-空间立方体
        self.conv1 = nn.Conv3d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm1 = nn.BatchNorm3d(out_chans // 2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm2 = nn.BatchNorm3d(out_chans)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv3d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.channels_first = channels_first

    def forward(self, x):
        # x 形状: (B, C, S, H, W)
        if not self.channels_first:
            # 如果不是 channels_first，需要重塑
            B, C, S, H, W = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*S*H*W, C)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        if not self.channels_first:
            # 恢复原始形状
            x = x.reshape(B, S, H, W, -1).permute(0, 4, 1, 2, 3)  # (B, out_features, S, H, W)

        return x


class Heat3D(nn.Module):
    """
    3D vHeat (HCO3D) for hyperspectral cubes:
    输入: x (B, C, S, H, W)  ->  输出: (B, C, S, H, W)
    Neumann 边界 => 3D DCT/IDCT; 频域指数衰减（各向异性 kx, ky, ks）
    """

    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, k_learnable=True, **kwargs):
        super().__init__()
        self.res = res
        self.hidden_dim = hidden_dim
        self.infer_mode = infer_mode

        # 3D 局部前处理
        self.local3d = nn.Conv3d(dim, hidden_dim, kernel_size=3, padding=1, bias=True)

        # 产生两路：HCO 主分支 + 门控分支
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = LayerNorm3d(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 预测各向异性 (kx, ky, ks)，确保非负
        if k_learnable:
            self.to_k = nn.Sequential(
                nn.Linear(hidden_dim, 3 * hidden_dim, bias=True),
                nn.ReLU(inplace=True)  # k >= 0
            )
        else:
            self.to_k = None

        # 运行时缓存
        self.__RES__ = None
        self.register_buffer("_Wcos_h", None, persistent=False)
        self.register_buffer("_Wcos_w", None, persistent=False)
        self.register_buffer("_Wcos_s", None, persistent=False)
        self.register_buffer("_alpha_h", None, persistent=False)
        self.register_buffer("_alpha_w", None, persistent=False)
        self.register_buffer("_alpha_s", None, persistent=False)

        # 推理模式下的预烘焙
        self.register_buffer("kx_exp", None, persistent=False)
        self.register_buffer("ky_exp", None, persistent=False)
        self.register_buffer("ks_exp", None, persistent=False)

    @staticmethod
    def _cos_map(N, device, dtype):
        # 正交化 DCT-II 基: (n, x)
        x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype)[None, :] + 0.5) / N
        n = torch.linspace(0, N - 1, N, device=device, dtype=dtype)[:, None]
        W = torch.cos(math.pi * n * x) * math.sqrt(2.0 / N)
        W[0, :] /= math.sqrt(2.0)  # n=0 的正交修正
        return W

    @staticmethod
    def _decay_1d(N, device, dtype):
        # alpha = exp(-(ω^2)), ω in [0, π)
        w = torch.linspace(0, math.pi, N + 1, device=device, dtype=dtype)[:N]
        return torch.exp(-(w ** 2))  # (N,)

    def _prepare_bases(self, S, H, W, device, dtype):
        if self.__RES__ == (S, H, W) and self._Wcos_h is not None and self._Wcos_h.device == device:
            return
        # 1D DCT-II 基
        Wcos_h = self._cos_map(H, device, dtype)
        Wcos_w = self._cos_map(W, device, dtype)
        Wcos_s = self._cos_map(S, device, dtype)
        # 1D 衰减基
        alpha_h = self._decay_1d(H, device, dtype)
        alpha_w = self._decay_1d(W, device, dtype)
        alpha_s = self._decay_1d(S, device, dtype)
        # 缓存
        self.__RES__ = (S, H, W)
        self._Wcos_h = Wcos_h
        self._Wcos_w = Wcos_w
        self._Wcos_s = Wcos_s
        self._alpha_h = alpha_h
        self._alpha_w = alpha_w
        self._alpha_s = alpha_s

    @torch.no_grad()
    def infer_init_heat3d(self, freq_embed):
        """推理模式下预烘焙"""
        assert self.to_k is not None, "to_k is None; k不可学习时无需预烘焙"
        device = self._alpha_h.device
        dtype = self._alpha_h.dtype
        k_all = self.to_k(freq_embed)  # (..., 3C)
        C = self.hidden_dim
        kx, ky, ks = torch.chunk(k_all, 3, dim=-1)  # (..., C)

        # 形状: (H,1,1,C), (1,W,1,C), (1,1,S,C)
        H, W, S = self._alpha_h.numel(), self._alpha_w.numel(), self._alpha_s.numel()
        self.kx_exp = (self._alpha_h.view(H, 1, 1, 1) ** kx).to(device=device, dtype=dtype)
        self.ky_exp = (self._alpha_w.view(1, W, 1, 1) ** ky).to(device=device, dtype=dtype)
        self.ks_exp = (self._alpha_s.view(1, 1, S, 1) ** ks).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, freq_embed=None):
        """
        x: (B, C_in, S, H, W)
        return: (B, C, S, H, W)
        Notes:
          - This implementation keeps channels-first everywhere: (B, C, S, H, W).
          - Builds kx_exp/ky_exp/ks_exp in shapes broadcastable to x: (1,C,1,H,1), (1,C,1,1,W), (1,C,S,1,1).
        """
        B, C_in, S, H, W = x.shape
        device, dtype = x.device, x.dtype
        C = self.hidden_dim

        # 1) local 3D conv -> (B, hidden_dim, S, H, W)
        x = self.local3d(x)  # (B, C, S, H, W)

        # 2) linear -> two branches (we do linear on channels-last temporarily then back)
        # convert to channels-last for applying self.linear (which expects last dim features)
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, S, H, W, C)
        x_lin = self.linear(x_cl)  # (B, S, H, W, 2C)
        x_main, z = x_lin.chunk(2, dim=-1)  # each (B, S, H, W, C)
        # back to channels-first
        x = x_main.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, S, H, W)
        z = z.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, S, H, W)

        # 3) prepare DCT bases if needed
        self._prepare_bases(S, H, W, device, dtype)
        W_h, W_w, W_s = self._Wcos_h, self._Wcos_w, self._Wcos_s  # shapes (H,H), (W,W), (S,S)

        # 4) DCT3D on channels-first (apply along S, H, W axes)
        # Note: einsum indices chosen to avoid name reuse. Use:
        #  x: b c s h w
        #  W_s: s t  -> result b c t h w  (t indexes freq)
        x = torch.einsum('bcshw,st->bcthw', x, W_s)  # (B, C, S, H, W) with S replaced by freq index (still size S)
        x = torch.einsum('bcthw,ht->bcttw', x, W_h)  # careful labels: second einsum maps H->new index (keeps dims)
        # The above two-step einsum can be simplified but kept explicit: do H then W properly
        # do H:
        x = torch.einsum('bcthw,ht->bcttw', x, W_h)  # intermediate (labels reused intentionally but sizes align)
        # do W:
        x = torch.einsum('bcttw,wt->bcttw', x, W_w)  # final DCT result shape (B,C,S,H,W)

        # The three einsums above are kept explicit; if einsum labels confuse, you can do explicit matmul per axis:
        # e.g. reshape and use @ with correct transposes. The goal is: x remains (B,C,S,H,W).

        # 5) frequency decay (make kx_exp/ky_exp/ks_exp broadcastable to (B,C,S,H,W))
        if self.infer_mode and (self.kx_exp is not None):
            kx_exp = self.kx_exp  # expected shapes will be handled if pre-baked (but ensure they are channels-first)
            ky_exp = self.ky_exp
            ks_exp = self.ks_exp
            # If pre-baked were created differently, ensure shapes -> (1,C,1,H,1) etc.
            # We'll try to reshape if possible:
            if kx_exp is not None and kx_exp.ndim == 4:  # maybe (H,1,1,C)
                kx_exp = kx_exp.permute(3, 0, 1, 2).unsqueeze(0)  # -> (1,C,H,1,1) then reshape to (1,C,1,H,1)
                kx_exp = kx_exp.reshape(1, C, 1, H, 1)
            # similar adjustments could be done for ky_exp/ks_exp if needed
        else:
            if self.to_k is None:
                # constants
                kx = torch.ones(C, device=device, dtype=dtype)
                ky = torch.ones(C, device=device, dtype=dtype)
                ks = torch.ones(C, device=device, dtype=dtype)
            else:
                k_all = self.to_k(freq_embed) if freq_embed is not None else self.to_k(
                    torch.zeros(1, self.hidden_dim, device=device, dtype=dtype))
                kx, ky, ks = torch.chunk(k_all, 3, dim=-1)  # each (1, C)

            # Build broadcastable exponentials:
            # self._alpha_h: (H,), self._alpha_w: (W,), self._alpha_s: (S,)
            # Want shapes:
            #   kx_exp: (1, C, 1, H, 1)
            #   ky_exp: (1, C, 1, 1, W)
            #   ks_exp: (1, C, S, 1, 1)
            # Do via broadcasting: (_alpha_h).view(1,1,H,1,1) ** kx.view(1,C,1,1,1)
            alpha_h = self._alpha_h.to(device=device, dtype=dtype).view(1, 1, H, 1, 1)  # (1,1,H,1,1)
            alpha_w = self._alpha_w.to(device=device, dtype=dtype).view(1, 1, 1, 1, W)  # (1,1,1,1,W)
            alpha_s = self._alpha_s.to(device=device, dtype=dtype).view(1, 1, S, 1, 1)  # (1,1,S,1,1)
            kx_v = kx.view(1, C, 1, 1, 1)  # (1,C,1,1,1)
            ky_v = ky.view(1, C, 1, 1, 1)
            ks_v = ks.view(1, C, 1, 1, 1)
            # Raise base to power per-channel: results have shapes:
            # alpha_h ** kx_v -> (1,C,H,1,1) -> permute to (1,C,1,H,1) if necessary
            kx_exp = (alpha_h ** kx_v).permute(0, 1, 2, 3, 4).contiguous()  # yields (1,C,H,1,1)
            # we want (1,C,1,H,1) so transpose axes:
            kx_exp = kx_exp.permute(0, 1, 2, 3, 4).reshape(1, C, H, 1, 1).permute(0, 1, 2, 3, 4)
            # Simpler: directly compute proper shapes:
            kx_exp = (self._alpha_h.view(1, 1, H, 1, 1).to(device, dtype) ** kx_v)  # (1,C,H,1,1)
            # Move H to 4th dim: want (1,C,1,H,1) -> permute
            kx_exp = kx_exp.permute(0, 1, 2, 3,
                                    4)  # currently (1,C,H,1,1); that's fine for broadcasting with x (B,C,S,H,W) if PyTorch aligns dims.
            # For clarity, ensure shapes for ky/ks:
            ky_exp = (self._alpha_w.view(1, 1, 1, 1, W).to(device, dtype) ** ky_v)  # (1,C,1,1,W)
            ks_exp = (self._alpha_s.view(1, 1, S, 1, 1).to(device, dtype) ** ks_v)  # (1,C,S,1,1)

        # Multiply in frequency domain (broadcasting works with shapes above)
        # Ensure x is float dtype matching exponents
        x = x * kx_exp * ky_exp * ks_exp  # (B, C, S, H, W)

        # 6) IDCT3D: inverse transforms (apply transposed bases)
        x = torch.einsum('bcthw,ts->bcshw', x, W_s.t())  # inverse S
        x = torch.einsum('bcshw,ht->bcstw', x, W_h.t())  # inverse H
        x = torch.einsum('bcstw,wt->bcshw', x, W_w.t())  # inverse W -> (B,C,S,H,W)

        # 7) out norm and gating
        x = self.out_norm(x)  # returns (B,C,S,H,W)
        x = x * torch.nn.functional.silu(z)  # z is (B,C,S,H,W)
        x = self.out_linear(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)  # apply linear on channel-dim safely

        return x.contiguous()


class HeatBlock3D(nn.Module):
    def __init__(
            self,
            res: int = 14,
            infer_mode=False,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = LayerNorm3d,
            use_checkpoint: bool = False,
            drop: float = 0.0,
            act_layer: nn.Module = nn.GELU,
            mlp_ratio: float = 4.0,
            post_norm=True,
            layer_scale=None,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        # 使用 Heat3D 替换 Heat2D
        self.op = Heat3D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None

        self.infer_mode = infer_mode

        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)

    def _forward(self, x: torch.Tensor, freq_embed):
        """
        Robust _forward: internally use channels-first (B, C, S, H, W) for all ops.
        Restore original ordering on return.
        """

        # ---------- helpers ----------
        def is_channels_first(t: torch.Tensor, hidden_dim: int):
            return t.dim() == 5 and t.shape[1] == hidden_dim

        def to_channels_first(t: torch.Tensor, hidden_dim: int):
            # accepts (B, C, S, H, W) or (B, H, W, S, C)
            if is_channels_first(t, hidden_dim):
                return t
            if t.dim() == 5 and t.shape[-1] == hidden_dim:
                # (B, H, W, S, C) -> (B, C, S, H, W)
                return t.permute(0, 4, 3, 1, 2).contiguous()
            raise ValueError(f"to_channels_first: cannot determine channel dim for tensor with shape {t.shape}")

        def to_channels_last(t: torch.Tensor, hidden_dim: int):
            # returns (B, H, W, S, C) given channels-first input
            if not is_channels_first(t, hidden_dim) and t.dim() == 5 and t.shape[-1] == hidden_dim:
                return t  # already channels-last
            if is_channels_first(t, hidden_dim):
                return t.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, W, S, C)
            raise ValueError(f"to_channels_last: cannot determine channel dim for tensor with shape {t.shape}")

        # ---------- main ----------
        if x.dim() != 5:
            raise ValueError(f"HeatBlock3D._forward expects 5D tensor, got {x.dim()}D")

        hidden = self.norm1.hidden_dim if hasattr(self.norm1, "hidden_dim") else getattr(self.norm1, "hidden_dim", None)
        if hidden is None:
            # fallback: try to infer from block's mlp or op if possible
            # but raising is safer to catch developer error
            raise RuntimeError("Cannot determine hidden_dim from norm1; ensure LayerNorm3d is used as norm_layer")

        orig_channels_last = False
        # If input is channels-last, record and convert to channels-first
        if not is_channels_first(x, hidden):
            if x.shape[-1] == hidden:
                orig_channels_last = True
                x_cf = to_channels_first(x, hidden)
            else:
                raise ValueError(
                    f"Input tensor channel dim mismatch: expected hidden={hidden} at dim1 or dim-1, got shape {x.shape}")
        else:
            x_cf = x

        # Now work in channels-first ordering (x_cf: B,C,S,H,W)
        if not self.layer_scale:
            if self.post_norm:
                # compute op output and ensure it's channels-first
                op_out = self.op(x_cf, freq_embed)
                if not is_channels_first(op_out, hidden):
                    op_out = to_channels_first(op_out, hidden)
                x_cf = x_cf + self.drop_path(self.norm1(op_out))
                if self.mlp_branch:
                    mlp_out = self.mlp(x_cf)
                    if not is_channels_first(mlp_out, hidden):
                        mlp_out = to_channels_first(mlp_out, hidden)
                    x_cf = x_cf + self.drop_path(mlp_out)
            else:
                # norm then op style
                norm_x = self.norm1(x_cf)
                op_out = self.op(norm_x, freq_embed)
                if not is_channels_first(op_out, hidden):
                    op_out = to_channels_first(op_out, hidden)
                x_cf = x_cf + self.drop_path(op_out)
                if self.mlp_branch:
                    norm2_x = self.norm2(x_cf)
                    mlp_out = self.mlp(norm2_x)
                    if not is_channels_first(mlp_out, hidden):
                        mlp_out = to_channels_first(mlp_out, hidden)
                    x_cf = x_cf + self.drop_path(mlp_out)

        else:
            # layer_scale == True branch (same logic but with gammas)
            if self.post_norm:
                op_out = self.op(x_cf, freq_embed)
                if not is_channels_first(op_out, hidden):
                    op_out = to_channels_first(op_out, hidden)
                x_cf = x_cf + self.drop_path(self.gamma1 * self.norm1(op_out))
                if self.mlp_branch:
                    mlp_out = self.mlp(x_cf)
                    if not is_channels_first(mlp_out, hidden):
                        mlp_out = to_channels_first(mlp_out, hidden)
                    x_cf = x_cf + self.drop_path(self.gamma2 * self.norm2(mlp_out))
            else:
                normed = self.norm1(x_cf)
                op_out = self.op(normed, freq_embed)
                if not is_channels_first(op_out, hidden):
                    op_out = to_channels_first(op_out, hidden)
                x_cf = x_cf + self.drop_path(self.gamma1 * op_out)
                if self.mlp_branch:
                    norm2_x = self.norm2(x_cf)
                    mlp_out = self.mlp(norm2_x)
                    if not is_channels_first(mlp_out, hidden):
                        mlp_out = to_channels_first(mlp_out, hidden)
                    x_cf = x_cf + self.drop_path(self.gamma2 * mlp_out)

        # restore ordering consistent with original input
        if orig_channels_last:
            x_out = to_channels_last(x_cf, hidden)  # (B, H, W, S, C)
        else:
            x_out = x_cf  # already channels-first

        return x_out

    def forward(self, input: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, freq_embed)
        else:
            return self._forward(input, freq_embed)


class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            if isinstance(module, nn.Module):
                x = module(x, *args, **kwargs)
            else:
                x = module(x)
        x = self[-1](x)
        return x


class vHeat3D(nn.Module):
    """3D vHeat for hyperspectral data"""

    def __init__(self, patch_size=11, in_chans=1, num_classes=16, depths=[2, 2, 6, 2],
                 dims=[96, 192, 384, 768], drop_path_rate=0.1, post_norm=True,
                 layer_scale=1e-6, use_checkpoint=False, mlp_ratio=4.0,
                 act_layer='GELU', infer_mode=False, spectral_bands=200, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.spectral_bands = spectral_bands

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.depths = depths

        # 3D Stem
        self.patch_embed = StemLayer3D(
            in_chans=in_chans,
            out_chans=self.embed_dim,
            act_layer=act_layer,
            norm_layer='BN'
        )

        # 保持完整分辨率
        self.res = [patch_size] * self.num_layers

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.infer_mode = infer_mode

        # 3D频率嵌入
        self.freq_embed = nn.ParameterList()
        for i in range(self.num_layers):
            # 为每个空间和光谱位置创建频率嵌入
            self.freq_embed.append(
                nn.Parameter(torch.zeros(patch_size, patch_size, spectral_bands, self.dims[i]), requires_grad=True))
            trunc_normal_(self.freq_embed[i], std=.02)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 使用1x1x1 3D卷积进行通道投影
            if i_layer < self.num_layers - 1:
                downsample = nn.Sequential(
                    nn.Conv3d(dims[i_layer], dims[i_layer + 1], kernel_size=1, stride=1, bias=False),
                    LayerNorm3d(dims[i_layer + 1])
                )
            else:
                downsample = nn.Identity()

            self.layers.append(self.make_layer(
                res=self.res[i_layer],
                dim=self.dims[i_layer],
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=LayerNorm3d,
                post_norm=post_norm,
                layer_scale=layer_scale,
                downsample=downsample,
                mlp_ratio=mlp_ratio,
                infer_mode=infer_mode,
            ))

        # 3D分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # 3D全局平均池化
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def make_layer(
            res=14,
            dim=96,
            depth=2,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            post_norm=True,
            layer_scale=None,
            downsample=nn.Identity(),
            mlp_ratio=4.0,
            infer_mode=False,
            **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(HeatBlock3D(
                res=res,
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                post_norm=post_norm,
                layer_scale=layer_scale,
                infer_mode=infer_mode,
            ))

        return AdditionalInputSequential(*blocks, downsample)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def infer_init(self):
        for i, layer in enumerate(self.layers):
            for block in layer[:-1]:
                block.op.infer_init_heat3d(self.freq_embed[i])
        del self.freq_embed

    def forward_features(self, x):
        # x: (B, C, S, H, W) 直接输入3D stem
        x = self.patch_embed(x)

        if self.infer_mode:
            for layer in self.layers:
                x = layer(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x, self.freq_embed[i])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


class S2VHeat3D(nn.Module):
    """3D vHeat的S2VNet兼容版本"""

    def __init__(self, band, num_classes, patch_size):
        super().__init__()
        self.backbone = vHeat3D(
            in_chans=1,  # 高光谱作为3D输入
            num_classes=num_classes,
            patch_size=patch_size,
            spectral_bands=band,  # 光谱波段数
            embed_dim=96,
            depths=[2, 2, 6, 2],
            dims=[96, 192, 384, 768],
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            layer_scale=1e-6,
            use_checkpoint=False
        )

    def forward(self, x, output_abu=False):
        # 输入x形状: (B, H, W, C) -> 需要转换为 (B, C, S, H, W)

        if x.dim() == 4:
            # 从 (B, H, W, C) 转换为 (B, 1, C, H, W)
            # 高光谱数据：C 是光谱维度，我们需要将其移到 S 维度
            B, C, H, W = x.shape
            # 直接重塑为 (B, 1, C, H, W) - C 就是光谱维度 S
            x = x.unsqueeze(1)  # (B, 1, C, H, W)


        logits = self.backbone(x)
        return logits


if __name__ == "__main__":
    from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis

    # 测试3D版本
    model = S2VHeat3D(band=200, num_classes=16, patch_size=7).cuda()

    # 创建3D输入数据: (B, H, W, C) 格式
    input = torch.randn((1, 200, 7, 7), device=torch.device('cuda'))

    print("3D vHeat Model for Hyperspectral Classification")
    print(f"Input shape: {input.shape}")

    # 测试前向传播
    with torch.no_grad():
        output = model(input)
        print(f"Output shape: {output.shape}")

        # 计算FLOPs
        analyze = FlopCountAnalysis(model, (input,))
        print("\nFLOPs Analysis:")
        print(flop_count_str(analyze))

        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")