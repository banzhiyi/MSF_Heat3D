import math
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

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


class FreqBranchAttention(nn.Module):
    """
    对多路 Heat3D 分支做频率自适应融合:
    输入: list of Tensors, 每个形状 [B, C, S, H, W]
    输出: 融合后的 [B, C, S, H, W]
    """
    def __init__(self, channels: int, num_branches: int = 3, alpha: float = 0.4,
                 hidden: int = None):
        super().__init__()
        self.channels = channels
        self.num_branches = num_branches
        hidden = hidden or channels
        self.alpha = alpha  # \* 新增: 学习注意力和均匀权重的插值系数
        # 输入为 [B, num_branches * C] 的全局频谱池化向量
        self.mlp = nn.Sequential(
            nn.Linear(num_branches * channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_branches)
        )

    def forward(self, feats):
        """
        feats: list of length num_branches, 每个 [B, C, S, H, W]
        """
        assert len(feats) == self.num_branches
        B, C, S, H, W = feats[0].shape

        # 1) 对每个分支做频谱+空间池化 -> [B, C]
        pooled_list = []
        for x in feats:
            # 先在 (S,H,W) 上做全局平均 -> [B, C]
            pooled = x.mean(dim=(2, 3, 4))
            pooled_list.append(pooled)
        # [B, num_branches*C]
        pooled_cat = torch.cat(pooled_list, dim=1)

        # 2) 通过 MLP 产生每个样本的 branch 权重 -> [B, num_branches]
        logits = self.mlp(pooled_cat)
        weights = F.softmax(logits, dim=-1)  # [B, num_branches]

        # \* 新增: 与均匀权重做温和插值，避免一开始注意力过于极端
        if self.alpha < 1.0:
            # 均匀权重 [1/num_branches, ..., 1/num_branches]
            uniform = torch.full_like(weights, 1.0 / self.num_branches)
            weights = self.alpha * weights + (1.0 - self.alpha) * uniform
            # 再归一化一次，确保行和为 1
            weights = weights / weights.sum(dim=-1, keepdim=True)

        # 3) 融合各分支
        out = 0.0
        for i, x in enumerate(feats):
            w = weights[:, i].view(B, 1, 1, 1, 1)
            out = out + w * x
        return out

class Heat3D(nn.Module):
    """
    3D vHeat (HCO3D) for hyperspectral cubes:
    输入: x (B, C, S, H, W)  ->  输出: (B, C, S, H, W)
    Neumann 边界 => 3D DCT/IDCT; 频域指数衰减（各向异性 kx, ky, ks）
    """

    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96,
                 k_learnable=True, use_local3d: bool = True,
                 light_in_proj: bool = False,use_multiscale: bool = False,
                 freq_mode: str = "mid",
                 **kwargs):
        super().__init__()
        self.res = res
        self.hidden_dim = hidden_dim
        self.input_dim = dim  # 新增：保存输入维度
        self.infer_mode = infer_mode
        self.use_multiscale = use_multiscale
        self.freq_mode = freq_mode  # \* 保存频率模式: "low" / "mid" / "high"
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

    def forward(self, x: torch.Tensor, freq_embed=None, freq_mask: torch.Tensor = None):
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

        # ====== 频带掩码：让不同 Heat3D 分支专责不同频段 ======
        # freq_mask 预期形状: (1,1,S,H,W) 或 (B,1,S,H,W)
        if freq_mask is not None:
            # 若只给了 (1,1,S,H,W)，这里 broadcast 到 (B,1,S,H,W)
            if freq_mask.shape[0] == 1 and B > 1:
                freq_mask = freq_mask.expand(B, -1, -1, -1, -1)
            # 若通道维为 1，这里再 broadcast 到 C 通道
            if freq_mask.shape[1] == 1 and C > 1:
                freq_mask = freq_mask.expand(-1, C, -1, -1, -1)
            # 最终与 x 同形状 \[B,C,S,H,W] 做逐元素乘法
            x = x * freq_mask

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
                # \* 根据 freq_mode 对 k 进行简单的频率偏置
                if self.freq_mode == "low":
                    scale = 1.2
                    kx = kx * scale
                    ky = ky * scale
                    ks = ks * scale

                elif self.freq_mode == "high":

                    # 减弱衰减：整体缩小 k，同时给一点偏置，避免全 0

                    scale = 0.9
                    bias = 0.05
                    kx = kx * scale + bias
                    ky = ky * scale + bias
                    ks = ks * scale + bias

                else:
                    # "mid" 或其他: 保持默认
                    pass

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

class ParallelHeat3DLayer(nn.Module):
    """
    第二层: 多个 Heat3D 分支并行 \+ 自适应频率融合
    假设所有分支输入输出通道相同: \[B, C, S, H, W]
    """
    def __init__(self, heat_block_cls, channels: int, inner_dim: int = None,
                 num_branches: int = 3,
                 freq_config: str = "low_mid_high",
                 parallel_cfg: Optional[dict] = None,
                 ):
        """
        freq_config:
          - "all_mid": \["mid", "mid", ...]
          - "mid_high": 例如 2 分支时 \["mid","high"]
          - "low_mid_high": 3 分支时 \["low","mid","high"]
        """
        super().__init__()
        self.num_branches = num_branches
        self.freq_config = freq_config  # 保存下来，便于核对
        inner_dim = inner_dim or channels // 2
        # 保存配置（带默认值）
        cfg = parallel_cfg or {}
        self.low_thresh = cfg.get("low_thresh", 0.33)
        self.high_thresh = cfg.get("high_thresh", 0.66)
        # 🆕 中频阈值
        self.mid_low = cfg.get("mid_low", 0.2)
        self.mid_high = cfg.get("mid_high", 0.8)
        self.eps_low = cfg.get("eps_low", 0.4)
        self.eps_mid = cfg.get("eps_mid", 0.3)
        self.eps_high = cfg.get("eps_high", 0.4)
        alpha = cfg.get("alpha", 0.5)
        # 根据策略生成 freq_modes
        if freq_config == "all_mid":
            base_freq_modes = ["mid"] * num_branches
        elif freq_config == "mid_high":
            # 逐步打开: 先只有 mid \+ high
            base_freq_modes = ["mid", "high", "high"]
        elif freq_config == "low_mid_high":
            base_freq_modes = ["low", "mid", "high"]
        elif freq_config == "low_high":
            base_freq_modes = ["low", "high", "high"]
        else:
            raise ValueError(f"未知 freq_config: {freq_config}")

        freq_modes = base_freq_modes[:num_branches]
        self.freq_modes = freq_modes  # ✅ 关键修复: 保存为成员变量，供 forward 使用
        self.branches = nn.ModuleList()
        for m in freq_modes:
            self.branches.append(
                heat_block_cls(
                    dim=channels,
                    hidden_dim=inner_dim,
                    use_local3d=True,
                    light_in_proj=False,
                    use_multiscale=True,
                    freq_mode=m
                )
            )

        # 把 inner_dim 投回 channels
        self.proj = nn.Conv3d(inner_dim, channels, kernel_size=1, bias=True)
        self.fuse = FreqBranchAttention(channels=inner_dim, num_branches=num_branches, alpha=alpha,)

    def forward(self, x, freq_embed_parallel=None):
        # x: (B, C, S, H, W)
        B, C, S, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # ----- 构造 1D 归一化频率索引 [0,1] -----
        idx_s = torch.linspace(0.0, 1.0, steps=S, device=device, dtype=dtype)  # (S,)
        idx_h = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=dtype)  # (H,)
        idx_w = torch.linspace(0.0, 1.0, steps=W, device=device, dtype=dtype)  # (W,)

        # 这里给一个简单的「半径型」频率定义：距离 0 位置的归一化距离越大，认为频率越高
        # freq_radius \[S,H,W] in [0,1]
        grid_s = idx_s.view(S, 1, 1)     # (S,1,1)
        grid_h = idx_h.view(1, H, 1)     # (1,H,1)
        grid_w = idx_w.view(1, 1, W)     # (1,1,W)
        # 一个简单合成：取三者的均值当作总频率
        freq_radius = (grid_s + grid_h + grid_w) / 3.0  # (S,H,W)

        # ----- 设计三个频带掩码: 低 / 中 / 高 -----
        # 阈值可以以后再调，这里先给一个直观划分:
        #   低频:   r in [0.0, 0.33]
        #   中频:   r in (0.2, 0.8)
        #   高频:   r in [0.66, 1.0]
        low_thresh = self.low_thresh
        high_thresh = self.high_thresh
        mid_low = self.mid_low
        mid_high = self.mid_high

        # 基础掩码
        mask_low = (freq_radius <= low_thresh).float()            # 低频为 1，其余为 0
        mask_high = (freq_radius >= high_thresh).float()          # 高频为 1，其余为 0
        # 中频：抑制最中心低频和最尖锐高频，仅保留中段
        mask_mid = ((freq_radius > mid_low) & (freq_radius < mid_high)).float()

        # 为了避免过于生硬，可以给被抑制部分一个小系数（而不是严格 0）
        eps_low = self.eps_low
        eps_mid = self.eps_mid
        eps_high = self.eps_high
        mask_low = eps_low + (1.0 - eps_low) * mask_low     # 低频区域 ~1，其余 ~eps_low
        mask_mid = eps_mid + (1.0 - eps_mid) * mask_mid     # 中频区域 ~1，其余 ~eps_mid
        mask_high = eps_high + (1.0 - eps_high) * mask_high # 高频区域 ~1，其余 ~eps_high

        # reshape 成 (1,1,S,H,W)，后面在 Heat3D 内 broadcast 到 (B,C,S,H,W)
        mask_map = {
            "low": mask_low.view(1, 1, S, H, W),
            "mid": mask_mid.view(1, 1, S, H, W),
            "high": mask_high.view(1, 1, S, H, W),
        }

        # ----- 动态跑 num_branches 个分支 -----
        feats = []
        for i, mode in enumerate(self.freq_modes):
            freq_mask = mask_map[mode]
            feats.append(self.branches[i](x, freq_embed_parallel, freq_mask=freq_mask))

        # 频率自适应融合，仍然是 \[B, inner_dim, S, H, W]
        out = self.fuse(feats)  # [B, inner_dim, S, H, W]

        # 使用 1x1x1 Conv3d 把 inner_dim 投回 channels，保持与输入一致
        out = self.proj(out)  # 形状变为 \[B, channels, S, H, W]

        return out

class MSF_Heat3D(nn.Module):
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
                 head_channels: int = 128,
                 reducer_type: str = "learnable",
                 pca_P: Optional[torch.Tensor] = None,
                 use_checkpoint: bool = False,
                 freq_pool: str = "avgmax",
                 use_post_norm: bool = True,
                 use_multiscale: bool = True,
                 dataset_name: str = "indian",
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

        ds = dataset_name.lower()
        if ds == "indian":
            # Indian 专用 freq 掩码 & alpha 配置
            self.parallel_cfg = {
                "low_thresh": 0.33,
                "high_thresh": 0.66,
                "mid_low": 0.2,  # 中频下界
                "mid_high": 0.8,  # 中频上界
                "eps_low": 0.4,
                "eps_mid": 0.3,
                "eps_high": 0.4,
                "alpha": 0.5,
            }
        elif ds in ["augsburg", "houston"]:
            # Augsburg 专用 freq 掩码 & alpha 配置（示例）
            self.parallel_cfg = {
                "low_thresh": 0.25,
                "high_thresh": 0.75,
                "mid_low": 0.3,  # 中频下界
                "mid_high": 0.7,  # 中频上界
                "eps_low": 0.4,
                "eps_mid": 0.3,
                "eps_high": 0.4,
                "alpha": 0.5,
            }
        elif dataset_name.lower() == "pavia":
            # Augsburg 专用 freq 掩码 & alpha 配置（示例）
            self.parallel_cfg = {
                "low_thresh": 0.25,
                "high_thresh": 0.75,
                "mid_low": 0.3,
                "mid_high": 0.7,
                "eps_low": 0.4,
                "eps_mid": 0.3,
                "eps_high": 0.4,
                "alpha": 0.3,
            }
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        # ---- 2) Heat3D 主干：第一层串行 + 第二层并行 ----
        # 第一层: 单个 Heat3D, 输入通道=1, 输出=heat_hidden_dim
        self.heat_first = Heat3D(
            dim=1,
            hidden_dim=heat_hidden_dim,
            use_local3d=True,
            light_in_proj=False,
            use_multiscale=use_multiscale,
            freq_mode="mid"  # 第一层用中频/常规模式即可
        )
        self.first_norm = LayerNorm3d(heat_hidden_dim) if use_post_norm else nn.Identity()

        # 第二层: 并行 Heat3D 分支
        self.heat_parallel = ParallelHeat3DLayer(
            heat_block_cls=Heat3D,
            channels=heat_hidden_dim,
            num_branches=3,
            freq_config="low_mid_high", # “low" "mid" "high"
            parallel_cfg=self.parallel_cfg,  #关键：把数据集专属配置传进去
        )
        self.second_norm = LayerNorm3d(heat_hidden_dim) if use_post_norm else nn.Identity()



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

    def forward(self, x: torch.Tensor, return_feat: bool = False) -> torch.Tensor:
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

        # 2) 第一层 Heat3D
        if self.use_checkpoint:
            x3d = torch.utils.checkpoint.checkpoint(self.heat_first, x3d, None)
        else:
            x3d = self.heat_first(x3d, None)
        x3d = self.first_norm(x3d)

        # 3) 第二层并行 Heat3D + 频率自适应融合
        residual = x3d
        if self.use_checkpoint:
            x3d = torch.utils.checkpoint.checkpoint(self.heat_parallel, x3d)
        else:
            x3d = self.heat_parallel(x3d)
        x3d = self.second_norm(x3d)  # [B, C, S, H, W]

        lambda_scale = 0.3  # 可以先写死一个小系数
        x3d = residual + lambda_scale * x3d

        # 频率融合 -> 2D Head
        x3d = self.spectral_fusion(x3d)  # (B, head_channels, S, H, W)
        x2d = self._spectral_pool(x3d)   # (B, C2d, H, W)，内部会做 freq_fuse
        #logits = self.head(x2d)          # (B, num_classes)
        #return logits
        # 复用 head 的前半段提取 logits 前向量特征
        # head: Conv2d -> BN -> GELU -> AdaptiveAvgPool2d(1) -> Flatten -> Linear
        feat = x2d
        feat = self.head[0](feat)
        feat = self.head[1](feat)
        feat = self.head[2](feat)
        feat = self.head[3](feat)
        feat_vec = self.head[4](feat)  # (B, head_channels)

        logits = self.head[5](feat_vec)  # Linear

        if return_feat:
            return logits, feat_vec
        return logits