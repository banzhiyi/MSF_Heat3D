import torch
import torch.nn as nn
from functools import partial
import math

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    print("Warning: mamba_ssm not installed. Using dummy Mamba.")
    Mamba = None
    RMSNorm = nn.LayerNorm
    layer_norm_fn = None
    rms_norm_fn = None

from GraphMamba.GCN import GCN


class Block(nn.Module):
    """Mamba Block with residual connection"""

    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
                 residual_in_fp32=False, drop_path=0.):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = nn.Identity()  # 简化版，可添加 DropPath

    def forward(self, hidden_states, residual=None, inference_params=None):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            # 使用融合操作（需要 mamba_ssm）
            if rms_norm_fn is not None:
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
                if residual is None:
                    hidden_states, residual = fused_add_norm_fn(
                        hidden_states, self.norm.weight, self.norm.bias,
                        residual=residual, prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                        eps=self.norm.eps,
                    )
                else:
                    hidden_states, residual = fused_add_norm_fn(
                        self.drop_path(hidden_states), self.norm.weight, self.norm.bias,
                        residual=residual, prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                        eps=self.norm.eps,
                    )

        #hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        # 修复：兼容 nn.Linear / Mamba 等不同 mixer 签名
        if inference_params is not None:
            try:
                hidden_states = self.mixer(hidden_states, inference_params=inference_params)
            except TypeError:
                hidden_states = self.mixer(hidden_states)
        else:
            hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False,
                 residual_in_fp32=False, fused_add_norm=False,
                 layer_idx=None, device=None, dtype=None):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    if Mamba is None:
        # Fallback: 使用简单的 Linear 代替 Mamba
        print("Warning: Using Linear layer instead of Mamba")
        mixer_cls = partial(nn.Linear, d_model, d_model, **factory_kwargs)
    else:
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon, **factory_kwargs
    )

    block = Block(
        d_model, mixer_cls, norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(module, n_layer, initializer_range=0.02,
                  rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class GraphMambaClassifier(nn.Module):
    """
    完整的 GraphMamba 实现，包含 Mamba 层和 GCN 层的双分支结构
    """

    def __init__(
            self,
            band: int,  # 输入波段数
            num_classes: int,  # 分类类别数
            patch_size: int,  # patch 大小（如 11）
            depth: int = 3,  # Mamba + GCN 层数
            embed_dim: int = 64,  # 嵌入维度
            gcn_layers: int = 3,  # 每个 GCN 分支内部的层数
            norm_epsilon: float = 1e-5,
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            drop_path_rate: float = 0.1,
            ssm_cfg: dict = None,
            use_spectral_adj: bool = True,  # 是否使用基于光谱的邻接矩阵
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_nodes = patch_size * patch_size
        self.use_spectral_adj = use_spectral_adj

        # 1. Patch embedding（光谱维度 → 嵌入维度）
        self.patch_to_embedding = nn.Linear(band, embed_dim)

        # 2. Mamba 层（状态空间模型）
        factory_kwargs = {"device": None, "dtype": None}
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                **factory_kwargs,
            )
            for i in range(depth)
        ])

        # 3. GCN 层（图结构学习）
        self.layer_GCN = nn.ModuleList([
            GCN(
                height=patch_size,
                width=patch_size,
                changel=embed_dim,
                layers_count=gcn_layers
            )
            for i in range(depth)
        ])

        # 4. Skip connections（深层特征融合）
        self.skipcat = nn.ModuleList([
            nn.Conv2d(self.n_nodes, self.n_nodes, [1, 2], 1, 0)
            for _ in range(depth - 2)
        ]) if depth > 2 else nn.ModuleList()

        # 5. 最终的 LayerNorm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon
        )

        # 6. 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        # 初始化权重
        self.apply(partial(_init_weights, n_layer=depth))

    def compute_adjacency(self, x: torch.Tensor, sigma: float = 10.0) -> torch.Tensor:
        """
        计算邻接矩阵
        x: (B, N, C) - patch 展平后的特征
        返回: (B, N, N) - 邻接矩阵
        """
        B, N, C = x.shape
        device = x.device

        if not self.use_spectral_adj:
            # 方案1：简单的网格邻接（4-邻域）
            A0 = self.build_grid_adj(self.patch_size, device)
            A = A0.unsqueeze(0).expand(B, -1, -1)
        else:
            # 方案2：基于光谱相似度的邻接矩阵（更接近原论文）
            # 计算节点间的欧氏距离
            prod = torch.bmm(x, x.transpose(1, 2))  # (B, N, N)
            norm = prod.diagonal(dim1=1, dim2=2).unsqueeze(2)  # (B, N, 1)
            dist = (norm + norm.transpose(1, 2) - 2 * prod).clamp(min=0)

            # 使用高斯核转换为相似度
            A = torch.exp(-dist / (sigma ** 2))

            # 可选：添加网格邻接作为先验
            A_grid = self.build_grid_adj(self.patch_size, device)
            A = A * A_grid.unsqueeze(0)  # 只保留网格邻域内的连接

        return A

    @staticmethod
    def build_grid_adj(patch_size: int, device: torch.device) -> torch.Tensor:
        """构建网格邻接矩阵（4-邻域）"""
        H = W = patch_size
        N = H * W
        A = torch.zeros((N, N), dtype=torch.float32, device=device)

        def idx(r, c):
            return r * W + c

        for r in range(H):
            for c in range(W):
                m = idx(r, c)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        n = idx(rr, cc)
                        A[m, n] = 1.0
        return A

    def forward_features(self, x: torch.Tensor, batch_A: torch.Tensor) -> torch.Tensor:
        """
        前向传播（特征提取）
        x: (B, N, C)
        batch_A: (B, N, N)
        """
        # Patch embedding
        hidden_states = self.patch_to_embedding(x)  # (B, N, embed_dim)

        # Mamba + GCN 双分支
        residual = None
        last_outputs = []

        for i, (mamba_block, gcn_block) in enumerate(zip(self.layers, self.layer_GCN)):
            # Skip connection（深层才启用）
            last_outputs.append(hidden_states)
            if i > 1:
                skip_input = torch.cat([
                    hidden_states.unsqueeze(3),
                    last_outputs[i - 2].unsqueeze(3)
                ], dim=3)
                hidden_states = self.skipcat[i - 2](skip_input).squeeze(3)

            # Mamba 分支（序列建模）
            hidden_states, residual = mamba_block(hidden_states, residual)

            # GCN 分支（图结构学习）
            hidden_states = gcn_block(hidden_states, batch_A)

        # 最终的 LayerNorm
        if residual is not None:
            residual = residual + hidden_states
        else:
            residual = hidden_states
        hidden_states = self.norm_f(residual)

        return hidden_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) 或 (B, H, W, C)
        返回: (B, num_classes)
        """
        # 统一输入格式
        if x.dim() == 4:
            if x.shape[1] <= 256:  # 假设 C <= 256（通道数在第2维）
                # (B, C, H, W) -> (B, H, W, C)
                x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            assert H == self.patch_size and W == self.patch_size
            x = x.reshape(B, H * W, C)
        else:
            B, N, C = x.shape
            assert N == self.n_nodes

        # 计算邻接矩阵
        batch_A = self.compute_adjacency(x)

        # 特征提取
        features = self.forward_features(x, batch_A)  # (B, N, embed_dim)

        # 全局���均池化
        pooled = features.mean(dim=1)  # (B, embed_dim)

        # 分类
        logits = self.head(pooled)  # (B, num_classes)

        return logits