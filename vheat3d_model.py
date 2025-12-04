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
def dct_1d(x, dim=-1):
    """
    对指定维度做 1D DCT-II，实数变换，线性、无参数。
    x: 任意形状 Tensor
    """
    N = x.size(dim)
    # 使用 FFT 的实数部分近似 DCT，保持线性
    # 这里采用一种常用 trick: 在 2N 上做 FFT，再取实部
    x = torch.cat([x, x.flip(dims=[dim])], dim=dim)  # 对称扩展
    X = torch.fft.rfft(x, dim=dim)
    # 只取前 N 个频率分量
    slices = [slice(None)] * x.dim()
    slices[dim] = slice(0, N)
    X = X[tuple(slices)].real
    return X


def idct_1d(X, dim=-1):
    """
    对指定维度做 1D 逆 DCT（对应上面的 dct_1d），保持线性。
    X: 任意形状 Tensor，最后一维为频率长度 N
    """
    N = X.size(dim)
    # 反向构造长度为 2N 的对称谱，然后用 irfft
    zeros_shape = list(X.shape)
    zeros_shape[dim] = 1
    zeros_pad = X.new_zeros(zeros_shape)

    # 拼出长度为 N+1 的 rfft 频谱（实数信号 rfft 长度为 N+1）
    # 这里用一个简化近似：补 0，然后 irfft，再截断
    X_rfft = torch.cat([X, zeros_pad], dim=dim)  # [ ..., N+1 ]
    x_rec = torch.fft.irfft(X_rfft, n=2 * N, dim=dim)

    # 取前 N 个样本作为近似的 idct 结果
    slices = [slice(None)] * x_rec.dim()
    slices[dim] = slice(0, N)
    x_rec = x_rec[tuple(slices)]
    return x_rec

class SpectralFrequencyGating(nn.Module):
    """
    仅在谱维做 1D 频域建模的轻量门控模块。

    预期输入/输出形状:
    - 输入:  x \[B, C, H, W, D\] 或 \[B, C, D, H, W\]，通过 `spec_dim` 控制谱维位置。
    - 输出: 同形状，乘上 \[B, C, 1, 1, D\] broadcast 的 sigmoid 门控。

    流程:
    1) H,W 上全局平均池化 -> x_mean \[B, C, D\]
    2) D 维上 DCT -> X_freq \[B, C, D\]
    3) 频域上 depthwise Conv1d -> X_freq_mod
    4) IDCT 回时域 -> gate_spec \[B, C, D\]
    5) sigmoid + broadcast -> x * gate
    """
    def __init__(self, channels, spec_length, spec_dim=-1, kernel_size=3, use_pointwise=True):
        super(SpectralFrequencyGating, self).__init__()
        self.channels = channels
        self.spec_length = spec_length
        self.spec_dim = spec_dim  # x 中谱维所在的维度索引

        padding = kernel_size // 2

        # 频域上的 depthwise Conv1d：每个通道一组频率响应
        self.dw_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=True
        )

        # 可选: 一个 pointwise Conv1d，在频率维上做轻量 mixing
        if use_pointwise:
            self.pw_conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                bias=True
            )
        else:
            self.pw_conv = None

    def forward(self, x):
        """
        x: 形状 \[B, C, ..., D, ...\]，其中谱维长度为 self.spec_length。
        只对谱维做 gating，不改变 H,W 结构。
        """
        # 1) 把谱维交换到最后，方便 pooling 和 DCT
        # 假设当前 spec_dim 位置是 self.spec_dim
        if self.spec_dim != -1:
            x = x.transpose(self.spec_dim, -1)  # 现在谱维在 -1

        # 此时假设 x: [B, C, H, W, D] 或 [B, C, *, D]
        B, C = x.shape[0], x.shape[1]
        D = x.shape[-1]

        # 2) 沿 H,W 做全局平均池化，只保留谱维 D
        #    无论中间有几个空间维，统统平均掉，只保留 [B, C, D]
        spatial_dims = list(range(2, x.dim() - 1))  # 排除 B,C,D 其余都视为空间维
        if len(spatial_dims) > 0:
            x_mean = x.mean(dim=spatial_dims, keepdim=False)  # [B, C, D]
        else:
            x_mean = x  # 已经没有空间维了

        # 3) DCT: 频域变换（线性，无参数）
        X_freq = dct_1d(x_mean, dim=-1)  # [B, C, D]

        # 4) 在频域做 1D conv：先视 C 为通道，用 Conv1d 的 (N=C, L=D) 约定
        #    Conv1d 期望输入 [B, C, L]，当前就是 [B, C, D]
        X_mod = self.dw_conv(X_freq)  # [B, C, D]
        if self.pw_conv is not None:
            X_mod = self.pw_conv(X_mod)  # [B, C, D]

        # 5) 逆 DCT 回谱域
        gate_spec = idct_1d(X_mod, dim=-1)  # [B, C, D]

        # 6) sigmoid 归一化，作为软门控
        gate_spec = torch.sigmoid(gate_spec)  # [B, C, D]

        # 7) 将 gate_spec broadcast 回原始 x 的形状
        #    先恢复谱维为 -1，其它空间维通过 unsqueeze/broadcast
        # 先扩展为 [B, C, 1, 1, D, ...] 与 x 匹配
        # 构造一个形状列表 [B, C, 1, 1, ..., D]
        while gate_spec.dim() < x.dim():
            gate_spec = gate_spec.unsqueeze(-2)  # 在 D 前面不断插入 1 维度

        # 现在 gate_spec 和 x 同维度数，最后一维都是 D，可以 broadcast
        x = x * gate_spec

        # 8) 如果一开始挪动过谱维位置，这里再挪回去
        if self.spec_dim != -1:
            x = x.transpose(self.spec_dim, -1)

        return x
# ---------- 谱降维模块 ----------
class LearnableSpectralReducer(nn.Module):
    """
    Learnable spectral reducer using a pointwise Conv1d over the spectral dimension.
    Input: (B, H, W, C)  or (B, C, H, W)
    Output: (B, S_red, H, W)
    """
    def __init__(self, in_bands:int, out_bands:int, use_bias:bool=True):
        super().__init__()
        self.in_bands = in_bands
        self.out_bands = out_bands
        # Conv1d that maps spectral channels -> reduced spectral channels.
        # Will be applied on (B, in_bands, H*W) via kernel_size=1
        self.conv1d = nn.Conv1d(in_channels=in_bands, out_channels=out_bands, kernel_size=1, bias=use_bias)
        # optional small BN + activation (helps stability)
        self.bn = nn.BatchNorm2d(out_bands)

    def forward(self, x):
        # Accept (B, H, W, C) or (B, C, H, W)
        if x.dim() == 4:  # (B, H, W, C)
            B, H, W, C = x.shape
            assert C == self.in_bands, f"in_bands mismatch: {C} vs {self.in_bands}"
            x = x.permute(0, 3, 1, 2).contiguous()  # -> (B, C, H, W)
        elif x.dim() == 4 and x.shape[1] == self.in_bands:
            pass
        elif x.dim() == 3:
            raise ValueError("unexpected 3D input")
        # now x is (B, C, H, W)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)          # (B, C, N) where N=H*W
        y = self.conv1d(x_flat)            # (B, S_red, N)
        y = y.view(B, self.out_bands, H, W) # (B, S_red, H, W)
        # optional BN+act (we keep channel as 'spectral channels', so use BN2d)
        y = self.bn(y)
        return y  # (B, S_red, H, W)

class PCASpectralReducer(nn.Module):
    """
    PCA-based reducer: offline compute projection matrix P (S_red x in_bands),
    then apply linear projection: y = P @ x_spectral at every pixel.
    Provide P as torch.tensor (S_red, in_bands).
    """
    def __init__(self, P:torch.Tensor):
        super().__init__()
        # P should be (S_red, in_bands)
        assert P.ndim == 2
        self.register_buffer("P", P.float())

    def forward(self, x):
        # x: (B, H, W, C) -> -> (B, S_red, H, W)
        if x.dim() == 4:
            B, H, W, C = x.shape
            x_perm = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        else:
            raise ValueError("PCASpectralReducer expects (B,H,W,C) input.")
        B, C, H, W = x_perm.shape
        x_flat = x_perm.view(B, C, -1)  # (B, C, N)
        # P (S_red, C) -> do batch matmul: y = P @ x_flat  => (B, S_red, N)
        y = torch.einsum("sc, bcn -> bsn", self.P, x_flat)
        y = y.view(B, self.P.shape[0], H, W)
        return y  # (B, S_red, H, W)

class LayerNorm3d(nn.Module):
    """3D LayerNorm that always returns channels-first ordering: (B, C, S, H, W).
    It accepts inputs in either (B, C, S, H, W) or (B, H, W, S, C).
    """

    def __init__(self, hidden_dim, eps=1e-6, debug=False):
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
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., channels_first=True):
        super().__init__()
        assert channels_first, "3D MLP must use channels_first=True for (B,C,S,H,W)"

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, bias=True)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B, C, S, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Heat3D(nn.Module):
    """
    3D vHeat (HCO3D) for hyperspectral cubes:
    输入: x (B, C, S, H, W)  ->  输出: (B, C, S, H, W)
    Neumann 边界 => 3D DCT/IDCT; 频域指数衰减（各向异性 kx, ky, ks）
    """

    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96,
                 k_learnable=True, use_local3d: bool = True,
                 light_in_proj: bool = False,use_multiscale: bool = False,
                 **kwargs):
        super().__init__()
        self.res = res
        self.hidden_dim = hidden_dim
        self.input_dim = dim  # 新增：保存输入维度
        self.infer_mode = infer_mode
        self.use_multiscale = use_multiscale

        # ----- 单尺度 / 多尺度 3D 局部前处理 -----
        if self.use_multiscale:
            # 多尺度 3 分支
            self.local_spec = nn.Conv3d(
                dim, hidden_dim,
                kernel_size=(1, 1, 3),
                stride=1,
                padding=(0, 0, 1),
                bias=True,
            )
            self.local_spat = nn.Conv3d(
                dim, hidden_dim,
                kernel_size=(3, 3, 1),
                stride=1,
                padding=(1, 1, 0),
                bias=True,
            )
            self.local_both = nn.Conv3d(
                dim, hidden_dim,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
                bias=True,
            )
            # 尺度权重门控：w = softmax(MLP(global_pool(x)))
            self.scale_mlp = nn.Sequential(
                nn.Linear(dim, dim, bias=True),
                nn.GELU(),
            )
            self.scale_proj = nn.Linear(dim, 3, bias=True)
        else:
            # 兼容旧逻辑：单一 local3d / 轻量 in_proj / Identity
            if use_local3d:
                self.local3d = nn.Conv3d(
                    dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
            else:
                if light_in_proj:
                    self.local3d = nn.Conv3d(
                        dim,
                        hidden_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                else:
                    if dim != hidden_dim:
                        raise ValueError(
                            f"Heat3D: use_local3d=False 且 light_in_proj=False 时, "
                            f"要求 dim == hidden_dim, 得到 dim={dim}, hidden_dim={hidden_dim}"
                        )
                    self.local3d = nn.Identity()

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
        Stable channels-first implementation.
        x: (B, C_in, S, H, W)
        return: (B, C, S, H, W)
        """
        B, C_in, S, H, W = x.shape
        device, dtype = x.device, x.dtype
        C = self.hidden_dim

        # 1) local conv / multi-scale local conv -> (B, C, S, H, W)
        if self.use_multiscale:
            # 三个尺度分支
            out1 = self.local_spec(x)  # (B, hidden_dim, S, H, W)
            out2 = self.local_spat(x)  # (B, hidden_dim, S, H, W)
            out3 = self.local_both(x)  # (B, hidden_dim, S, H, W)

            # 全局池化：在 (S,H,W) 上做 mean，得到 (B, dim)
            # 输入 x 仍是原始输入通道数 dim
            B, C_in, S, H, W = x.shape
            gp = x.mean(dim=(2, 3, 4))  # (B, C_in)

            # MLP -> 3 标量权重 (每个样本一组三尺度权重)
            scale_feat = self.scale_mlp(gp)  # (B, C_in)
            logits = self.scale_proj(scale_feat)  # (B, 3)
            weights = torch.softmax(logits, dim=-1)  # (B, 3)

            # reshape 权重以便广播到 5D 特征上
            w1 = weights[:, 0].view(B, 1, 1, 1, 1)
            w2 = weights[:, 1].view(B, 1, 1, 1, 1)
            w3 = weights[:, 2].view(B, 1, 1, 1, 1)

            x = w1 * out1 + w2 * out2 + w3 * out3  # (B, hidden_dim, S, H, W)
        else:
            x = self.local3d(x)  # (B, hidden_dim, S, H, W)

        # 2) Linear -> split into main & gate branches.
        # self.linear expects last-dim features -> temporarily move channels to last
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, S, H, W, C)
        x_lin = self.linear(x_cl)  # (B, S, H, W, 2C)
        x_main, z = x_lin.chunk(2, dim=-1)  # (B, S, H, W, C) each
        # back to channels-first
        x = x_main.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, S, H, W)
        z = z.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, S, H, W)

        # 3) prepare dct bases
        self._prepare_bases(S, H, W, device, dtype)
        W_h, W_w, W_s = self._Wcos_h, self._Wcos_w, self._Wcos_s  # shapes: (H,H),(W,W),(S,S)

        # ---------- Forward DCT along S, then H, then W ----------
        # --- transform S axis ---
        # bring S to last: (B,C,S,H,W) -> permute -> (B,C,H,W,S)
        x_perm = x.permute(0, 1, 3, 4, 2).contiguous()  # (B,C,H,W,S)
        x_2d = x_perm.view(-1, S)  # (B*C*H*W, S)
        # multiply by W_s^T to project: result (B*C*H*W, S)
        x_2d = x_2d @ W_s.t()  # (B*C*H*W, S)
        x_perm = x_2d.view(B, C, H, W, S)  # (B,C,H,W,S)
        x = x_perm.permute(0, 1, 4, 2, 3).contiguous()  # back to (B,C,S,H,W)

        # --- transform H axis ---
        x_perm = x.permute(0, 1, 2, 4, 3).contiguous()  # (B,C,S,W,H) bring H to last
        x_2d = x_perm.view(-1, H)  # (B*C*S*W, H)
        x_2d = x_2d @ W_h.t()  # (B*C*S*W, H)
        x_perm = x_2d.view(B, C, S, W, H)  # (B,C,S,W,H)
        x = x_perm.permute(0, 1, 2, 4, 3).contiguous()  # back to (B,C,S,H,W)

        # --- transform W axis ---
        x_2d = x.view(-1, W)  # (B*C*S*H, W) since W is last already
        x_2d = x_2d @ W_w.t()  # (B*C*S*H, W)
        x = x_2d.view(B, C, S, H, W)  # (B,C,S,H,W)

        # ---------- Frequency-domain damping (make shapes broadcastable) ----------
        # Build per-channel k vectors
        if self.to_k is None:
            kx = ky = ks = torch.ones(C, device=device, dtype=dtype)
        else:
            if self.to_k is None:
                # 常数 k
                kx = ky = ks = torch.ones(C, device=device, dtype=dtype)
            else:
                # 规范化 freq_embed 到 (1, C) 再输入 to_k
                if freq_embed is None:
                    fe = torch.zeros(1, self.hidden_dim, device=device, dtype=dtype)  # (1, C)
                else:
                    fe = freq_embed
                    # 如果最后一维是通道 dim（C），对其他维取均值
                    if fe.dim() >= 2 and fe.shape[-1] == self.hidden_dim:
                        # e.g. (H,W,S,C) or (S,C) or (H,W,C) -> average所有非通道维
                        # 计算需要保留最后一维，其他维取 mean
                        reduce_dims = tuple(range(fe.dim() - 1))  # e.g. (0,1,2)
                        fe = fe.mean(dim=reduce_dims)  # becomes (C,)
                        fe = fe.unsqueeze(0)  # -> (1, C)
                    elif fe.dim() == 1 and fe.numel() == self.hidden_dim:
                        fe = fe.unsqueeze(0)  # (1, C)
                    elif fe.dim() == 2 and fe.shape[1] == self.hidden_dim:
                        # already (N, C) -- keep as is, we'll take first row if N>1
                        if fe.shape[0] > 1:
                            fe = fe.mean(dim=0, keepdim=True)  # aggregate across batch-like dim
                    else:
                        # 最后兜底：尝试把张量展平并取前 C 个数作为特征（不常用）
                        fe = fe.reshape(1, -1)[:, :self.hidden_dim]
                # 现在 fe 应为 (1, C)
                k_all = self.to_k(fe)  # expected (1, 3C) or (3C,)

                # ---- 防 NaN / 极端值：对 k_all 做 nan_to_num + clamp ----
                k_all = torch.nan_to_num(k_all, nan=0.0, posinf=10.0, neginf=0.0)
                # k>=0 的设定下，主要限制上界；若后续需要可放宽
                k_all = torch.clamp(k_all, 0.0, 10.0)

                # 规范 k_all 到 1D 长向量 (3C,)
                if k_all.dim() == 2 and k_all.shape[0] == 1:
                    k_all = k_all.squeeze(0)
                elif k_all.dim() == 2 and k_all.shape[0] > 1:
                    # 多行情况：平均到一行
                    k_all = k_all.mean(dim=0)
                # 最终切分
                kx, ky, ks = torch.chunk(k_all, 3, dim=-1)
                # 确保为 1D 向量 (C,)
                kx = kx.reshape(-1)
                ky = ky.reshape(-1)
                ks = ks.reshape(-1)

        # ------------------ 使用 kx_eff, ky_eff, ks_eff 构造衰减因子 ------------------
        kx_v = kx.view(1, C, 1, 1, 1)
        ky_v = ky.view(1, C, 1, 1, 1)
        ks_v = ks.view(1, C, 1, 1, 1)

        alpha_h = self._alpha_h.to(device=device, dtype=dtype).view(1, 1, 1, H, 1)
        alpha_w = self._alpha_w.to(device=device, dtype=dtype).view(1, 1, 1, 1, W)
        alpha_s = self._alpha_s.to(device=device, dtype=dtype).view(1, 1, S, 1, 1)

        kx_exp = (alpha_h ** kx_v)
        ky_exp = (alpha_w ** ky_v)
        ks_exp = (alpha_s ** ks_v)

        x = x * kx_exp * ky_exp * ks_exp

        # ---------- Inverse DCT (IDCT) along W, H, S using W_.T reversed order ----------
        # inverse W
        x_2d = x.view(-1, W)  # (B*C*S*H, W)
        x_2d = x_2d @ W_w  # multiply by W_w (inverse)
        x = x_2d.view(B, C, S, H, W)

        # inverse H
        x_perm = x.permute(0, 1, 2, 4, 3).contiguous()  # (B,C,S,W,H)
        x_2d = x_perm.view(-1, H)  # (B*C*S*W, H)
        x_2d = x_2d @ W_h  # (B*C*S*W, H)
        x_perm = x_2d.view(B, C, S, W, H)
        x = x_perm.permute(0, 1, 2, 4, 3).contiguous()  # (B,C,S,H,W)

        # inverse S
        x_perm = x.permute(0, 1, 3, 4, 2).contiguous()  # (B,C,H,W,S)
        x_2d = x_perm.view(-1, S)  # (B*C*H*W, S)
        x_2d = x_2d @ W_s  # (B*C*H*W, S)
        x_perm = x_2d.view(B, C, H, W, S)
        x = x_perm.permute(0, 1, 4, 2, 3).contiguous()  # (B,C,S,H,W)

        # ---------- Output normalization, gating and linear ----------
        x = self.out_norm(x)  # LayerNorm3d expects channels-first -> returns (B,C,S,H,W)
        x = x * torch.nn.functional.silu(z)  # z is (B,C,S,H,W)
        # out_linear expects last-dim features -> temporarily move channels to last
        x = self.out_linear(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()

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
            # 新增瓶颈结构参数
            bottleneck_ratio: float = 0.25,
            use_bottleneck: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        # 新增：瓶颈结构
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck_dim = max(8, int(hidden_dim * bottleneck_ratio))
            # 压缩卷积
            self.bottleneck_compress = nn.Conv3d(
                hidden_dim, self.bottleneck_dim, kernel_size=1, stride=1, bias=True
            )
            self.bottleneck_norm = nn.BatchNorm3d(self.bottleneck_dim)
            self.bottleneck_act = act_layer()
            # 扩展卷积
            self.bottleneck_expand = nn.Conv3d(
                self.bottleneck_dim, hidden_dim, kernel_size=1, stride=1, bias=True
            )

        # 修改：正确传递维度给Heat3D
        op_input_dim = self.bottleneck_dim if self.use_bottleneck else hidden_dim
        op_hidden_dim = self.bottleneck_dim if self.use_bottleneck else hidden_dim
        # 使用 Heat3D 替换 Heat2D
        self.op = Heat3D(res=res, dim=op_input_dim, hidden_dim=op_hidden_dim, infer_mode=infer_mode)
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

        def ensure_op_out_matches(op_out: torch.Tensor, target_shape: tuple, hidden_dim: int):
            """
            Ensure op_out becomes channels-first with spatial order matching target_shape=(S,H,W).
            Returns op_out permuted to (B,C,S,H,W).
            """
            # 1) make channels-first if needed
            if not is_channels_first(op_out, hidden_dim):
                op_out = to_channels_first(op_out, hidden_dim)

            # 2) if spatial dims already match, return
            if op_out.shape[2:5] == target_shape:
                return op_out

            # 3) try permutations of spatial axes (indices 2,3,4)
            import itertools
            for perm in itertools.permutations((2, 3, 4)):
                # build full permute tuple: (0,1,p2,p3,p4)
                full_perm = (0, 1, perm[0], perm[1], perm[2])
                permuted = op_out.permute(full_perm).contiguous()
                if permuted.shape[2:5] == target_shape:
                    return permuted

            # 4) if not found, raise informative error
            raise RuntimeError(f"Cannot align op_out spatial dims {op_out.shape[2:5]} to target {target_shape}. "
                               "Tried channel-first conversion and all spatial permutations.")

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
        target_spatial = x_cf.shape[2:5]  # (S, H, W)
        # 新增：瓶颈压缩
        if self.use_bottleneck:
            x_compressed = self.bottleneck_compress(x_cf)
            x_compressed = self.bottleneck_norm(x_compressed)
            x_compressed = self.bottleneck_act(x_compressed)
        else:
            x_compressed = x_cf

        if not self.layer_scale:
            if self.post_norm:
                # compute op output and ensure it's channels-first AND spatially aligned to x_cf
                op_out = self.op(x_compressed, freq_embed)
                op_out = ensure_op_out_matches(op_out, target_spatial, self.bottleneck_dim if self.use_bottleneck else hidden)
                # 新增：瓶颈扩展
                if self.use_bottleneck:
                    op_out = self.bottleneck_expand(op_out)
                    # 扩展后需要再次确保空间维度匹配
                    op_out = ensure_op_out_matches(op_out, target_spatial, hidden)
                x_cf = x_cf + self.drop_path(self.norm1(op_out))

                if self.mlp_branch:
                    mlp_out = self.mlp(x_cf)
                    if not is_channels_first(mlp_out, hidden):
                        mlp_out = to_channels_first(mlp_out, hidden)
                    # mlp_out should already be channels-first with correct spatial order (it uses channels_first=True)
                    x_cf = x_cf + self.drop_path(mlp_out)
            else:
                # norm then op style
                norm_x = self.norm1(x_cf)
                # 瓶颈压缩
                if self.use_bottleneck:
                    norm_x_compressed = self.bottleneck_compress(norm_x)
                    norm_x_compressed = self.bottleneck_norm(norm_x_compressed)
                    norm_x_compressed = self.bottleneck_act(norm_x_compressed)
                else:
                    norm_x_compressed = norm_x
                op_out = self.op(norm_x_compressed, freq_embed)
                op_out = ensure_op_out_matches(op_out, target_spatial, self.bottleneck_dim if self.use_bottleneck else hidden)
                # 瓶颈扩展
                if self.use_bottleneck:
                    op_out = self.bottleneck_expand(op_out)
                    op_out = ensure_op_out_matches(op_out, target_spatial, hidden)
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
                # 瓶颈压缩
                if self.use_bottleneck:
                    x_compressed = self.bottleneck_compress(x_cf)
                    x_compressed = self.bottleneck_norm(x_compressed)
                    x_compressed = self.bottleneck_act(x_compressed)
                op_out = self.op(x_compressed, freq_embed)
                op_out = ensure_op_out_matches(op_out, target_spatial, self.bottleneck_dim if self.use_bottleneck else hidden)
                # 瓶颈扩展
                if self.use_bottleneck:
                    op_out = self.bottleneck_expand(op_out)
                    op_out = ensure_op_out_matches(op_out, target_spatial, hidden)
                x_cf = x_cf + self.drop_path(self.gamma1.view(1,-1,1,1,1) * self.norm1(op_out))

                if self.mlp_branch:
                    mlp_out = self.mlp(x_cf)
                    if not is_channels_first(mlp_out, hidden):
                        mlp_out = to_channels_first(mlp_out, hidden)
                    x_cf = x_cf + self.drop_path(self.gamma2.view(1,-1,1,1,1) * self.norm2(mlp_out))
            else:
                normed = self.norm1(x_cf)
                # 瓶颈压缩
                if self.use_bottleneck:
                    normed_compressed = self.bottleneck_compress(normed)
                    normed_compressed = self.bottleneck_norm(normed_compressed)
                    normed_compressed = self.bottleneck_act(normed_compressed)
                else:
                    normed_compressed = normed
                op_out = self.op(normed_compressed, freq_embed)
                op_out = ensure_op_out_matches(op_out, target_spatial, self.bottleneck_dim if self.use_bottleneck else hidden)
                op_out = ensure_op_out_matches(op_out, target_spatial, hidden)
                x_cf = x_cf + self.drop_path(self.gamma1.view(1,-1,1,1,1) * op_out)
                if self.mlp_branch:
                    norm2_x = self.norm2(x_cf)
                    mlp_out = self.mlp(norm2_x)
                    if not is_channels_first(mlp_out, hidden):
                        mlp_out = to_channels_first(mlp_out, hidden)
                    x_cf = x_cf + self.drop_path(self.gamma2.view(1,-1,1,1,1) * mlp_out)

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

    def __init__(self, patch_size=11, in_chans=1, num_classes=16, depths=[1, 1, 3, 1],
                 dims=[64, 128, 256, 512], drop_path_rate=0.1, post_norm=True,
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
                # 正确写法 — 保留最后一维为通道 C，形状 (H, W, S, C)
                freq_i = self.freq_embed[i]  # (H, W, S, C)
                x = layer(x, freq_i)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


class S2VHeat3D(nn.Module):
    """3D vHeat的S2VNet兼容版本"""

    def __init__(self, band, num_classes, patch_size, use_bottleneck=True, bottleneck_ratio=0.25):
        super().__init__()
        self.backbone = vHeat3D(
            in_chans=1,  # 高光谱作为3D输入
            num_classes=num_classes,
            patch_size=patch_size,
            spectral_bands=band,  # 光谱波段数
            embed_dim=64,
            depths=[1, 1, 3, 1],
            dims=[64, 128, 256, 512],
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            layer_scale=1e-6,
            use_checkpoint=False,
            # 启用瓶颈
            use_bottleneck=use_bottleneck,
            bottleneck_ratio=bottleneck_ratio,
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



class Heat3D_Pipeline(nn.Module):
    """
    (B, H, W, S) 或 (B, S, H, W) ->
      1) 光谱降维 (S -> S')
      2) Heat3D 堆叠 (首层 1->hidden_dim，后续 hidden_dim->hidden_dim)
      3) 频谱池化 -> 2D Head -> logits
    """
    def __init__(self,
                 band: int,
                 num_classes: int,
                 patches: int,
                 reduced_bands: int = 24,
                 heat_hidden_dim: int = 48,
                 n_heat_layers: int = 2,
                 head_channels: int = 128,
                 reducer_type: str = "learnable",
                 pca_P: Optional[torch.Tensor] = None,
                 use_checkpoint: bool = False,
                 freq_pool: str = "avgmax",
                 use_post_norm: bool = True,
                 use_multiscale: bool = True,
                 ):
        super().__init__()
        self.band = band
        self.num_classes = num_classes
        self.patches = patches
        self.reduced_bands = reduced_bands
        self.use_checkpoint = use_checkpoint
        self.freq_pool = freq_pool
        self.use_post_norm = use_post_norm
        # 1) 光谱降维器：其实现期望 (B, H, W, S)
        if reducer_type == "learnable":
            self.reducer = LearnableSpectralReducer(in_bands=band, out_bands=reduced_bands)
        elif reducer_type == "pca":
            assert pca_P is not None and isinstance(pca_P, torch.Tensor), "pca_P 不能为空"
            assert pca_P.ndim == 2, f"pca_P 期望 2D, 得到 {pca_P.shape}"
            assert pca_P.shape[0] == reduced_bands, (
                f"PCA 投影矩阵第一维({pca_P.shape[0]})应为 reduced_bands={reduced_bands}"
            )
            assert pca_P.shape[1] == band, (
                f"PCA 投影矩阵第二维({pca_P.shape[1]})应为 band={band}"
            )
            self.reducer = PCASpectralReducer(pca_P)
        else:
            raise ValueError(f"未知 reducer_type: {reducer_type}")

        # 2) Heat3D 堆叠：首层保留 3x3x3 local3d，后续层可改为轻量 1x1x1 卷积
        modules = []
        for i in range(n_heat_layers):
            in_dim = 1 if i == 0 else heat_hidden_dim

            # 只在第一层打开多尺度，后面保持单尺度
            enable_ms = use_multiscale

            modules.append(
                Heat3D(
                    dim=in_dim,
                    hidden_dim=heat_hidden_dim,
                    use_local3d=True,  # 仍然使用 3x3x3 作为其中一支 (local_both)
                    light_in_proj=False,
                    use_multiscale=enable_ms,
                )
            )

        self.heat_modules = nn.ModuleList(modules)

        # post_norm 改为可选
        if self.use_post_norm:
            self.post_norms = nn.ModuleList([LayerNorm3d(heat_hidden_dim) for _ in range(n_heat_layers)])
        else:
            self.post_norms = nn.ModuleList([nn.Identity() for _ in range(n_heat_layers)])

        # 3) 频率融合与分类头
        self.spectral_fusion = nn.Conv3d(
            in_channels=heat_hidden_dim, out_channels=head_channels,
            kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True
        )

        # 恢复：avgmax 只做简单拼接，不加 1x1 conv
        if freq_pool == "avgmax":
            out2d_channels = head_channels * 2
        else:
            out2d_channels = head_channels

        self.head = nn.Sequential(
            nn.Conv2d(out2d_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(head_channels, num_classes)
        )

    @staticmethod
    def _to_channels_last_hw_s(x: torch.Tensor, band: int) -> torch.Tensor:
        """
        接受 (B, H, W, S) / (B, S, H, W) / (B, W, H, S) 等，输出统一为 (B, H, W, S)
        更健壮地自动检测哪个维度是光谱维 (== band)。
        """
        if x.dim() != 4:
            raise ValueError(f"期望 4D 输入, 得到 {x.shape}")

        B, d1, d2, d3 = x.shape
        dims = [d1, d2, d3]

        # 找出与 band 相等的维度索引
        spectral_candidates = [i for i, d in enumerate(dims) if d == band]
        if len(spectral_candidates) == 0:
            raise ValueError(f"无法在 {x.shape} 中找到等于 band={band} 的光谱维")
        if len(spectral_candidates) > 1:
            raise ValueError(f"输入形状 {x.shape} 中有多个维度等于 band={band}, "
                             f"无法唯一确定光谱维, candidates={spectral_candidates}")

        s_idx = spectral_candidates[0]  # 0,1,2 分别对应原来的 dim1,2,3

        # 根据 s_idx 构造到 (B,H,W,S) 的 permute
        if s_idx == 2:
            # (B, H, W, S) 已经是目标格式
            return x.contiguous()
        elif s_idx == 0:
            # (B, S, H, W) -> (B, H, W, S)
            return x.permute(0, 2, 3, 1).contiguous()
        elif s_idx == 1:
            # (B, H, S, W) 或 (B, W, S, H)，需要再判断剩余两个维度
            remaining = [i for i in range(3) if i != s_idx]
            h_idx, w_idx = remaining
            # 默认 (B, H, S, W) -> (B, H, W, S)
            return x.permute(0, h_idx + 1, w_idx + 1, s_idx + 1).contiguous()
        else:
            # 不应该到这里
            raise RuntimeError(f"意外的 spectral index {s_idx} for shape {x.shape}")

    def _spectral_pool(self, x3d: torch.Tensor) -> torch.Tensor:
        """
        x3d: (B, C, S', H, W) -> (B, C2d , H, W)
        """
        if self.freq_pool == "avg":
            return x3d.mean(dim=2)
        if self.freq_pool == "max":
            return x3d.max(dim=2)[0]
        if self.freq_pool == "avgmax":
            avg = x3d.mean(dim=2)
            mx = x3d.max(dim=2)[0]
            return torch.cat([avg, mx], dim=1)
        raise ValueError(f"未知 freq_pool: {self.freq_pool}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 形状统一到 (B, H, W, S) 以匹配 reducer
        x_cl = self._to_channels_last_hw_s(x, self.band)
        B, H, W, S = x_cl.shape
        if H != self.patches or W != self.patches:
            raise AssertionError(f"patch 尺寸不匹配: {H}x{W} != {self.patches}")

        # 光谱降维: 期望输出 (B, S_red, H, W)
        x_red = self.reducer(x_cl)
        if x_red.dim() != 4:
            raise ValueError(f"reducer 输出应为 4D, 得到 {x_red.shape}")

        # ---- 检查 reducer 输出的光谱维是否与 reduced_bands 一致 ----
        # 允许两种主布置：通道在 dim=1 或 dim=-1，其余情况报错
        if x_red.shape[1] == self.reduced_bands:
            layout = "CHW"  # (B, S', H, W)
        elif x_red.shape[-1] == self.reduced_bands:
            layout = "HWCh"  # (B, H, W, S')
        else:
            raise AssertionError(
                f"reducer 输出光谱维与 reduced_bands 不一致: 输出 {x_red.shape}, "
                f"reduced_bands={self.reduced_bands}"
            )

        # 统一到 3D 输入 (B, 1, S', H, W)
        if layout == "CHW":            # (B, S', H, W)
            x3d = x_red.unsqueeze(1).contiguous()
        else:                           # (B, H, W, S')
            x3d = x_red.permute(0, 3, 1, 2).unsqueeze(1).contiguous()

        # Heat3D 堆叠：首层 1->hidden_dim，其后 hidden_dim->hidden_dim
        for module, norm in zip(self.heat_modules, self.post_norms):
            if self.use_checkpoint:
                x3d = checkpoint.checkpoint(module, x3d, None)
            else:
                x3d = module(x3d, None)
            # post_norm 可选
            x3d = norm(x3d)  # (B, hidden_dim, S, H, W)

        # 频率融合 -> 2D Head
        x3d = self.spectral_fusion(x3d)  # (B, head_channels, S, H, W)
        x2d = self._spectral_pool(x3d)   # (B, C2d, H, W)，内部会做 freq_fuse
        logits = self.head(x2d)          # (B, num_classes)
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