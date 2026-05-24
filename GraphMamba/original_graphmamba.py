"""
Original-style GraphMamba model adapter.

This module keeps the core architecture used by ahappyyang/GraphMamba:
VisionMamba blocks, one GCN branch after every Mamba block, skip fusion from
the third block onward, and classification from the center patch token.

The original training script feeds flattened patches plus precomputed graph
adjacency matrices.  For compatibility with this project, ``forward`` also
accepts regular HSI patches in ``(B, C, H, W)`` format and builds the same local
spectral graph on the fly.
"""

from functools import partial
import math
from typing import Optional
import warnings

import torch
from torch import Tensor
import torch.nn as nn

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class DropPath(nn.Module):
    """Drop paths per sample, matching timm's behavior for this use case."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class OriginalGraphMambaGCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))

    def A_to_D_inv(self, A: Tensor) -> Tensor:
        D = A.sum(2).clamp_min(1e-12)
        D_hat = torch.pow(D, -0.5)
        return torch.diag_embed(D_hat)

    def forward(self, H: Tensor, A: Tensor) -> Tensor:
        nodes_count = A.shape[1]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False, device=A.device, dtype=A.dtype)
        A = A + I
        batch, length, channels = H.shape
        H = self.BN(H.reshape(batch * length, channels)).reshape(batch, length, channels)
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        output = torch.matmul(A_hat, self.GCN_liner_out_1(H))
        return self.Activition(output)


class OriginalGraphMambaGCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, layers_count: int):
        super().__init__()
        self.channel = changel
        self.height = height
        self.width = width
        self.GCN_Branch = nn.Sequential()
        for i in range(layers_count):
            self.GCN_Branch.add_module(
                "GCN_Branch" + str(i),
                OriginalGraphMambaGCNLayer(self.channel, self.channel),
            )
        self.BN = nn.BatchNorm1d(64)

    def forward(self, x: Tensor, A: Tensor) -> Tensor:
        H = x
        for i in range(len(self.GCN_Branch)):
            H = self.GCN_Branch[i](H, A)
        return H


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            if RMSNorm is None:
                raise ImportError("fused_add_norm=True requires mamba_ssm Triton RMSNorm support.")
            if not isinstance(self.norm, (nn.LayerNorm, RMSNorm)):
                raise TypeError("Only LayerNorm and RMSNorm are supported for fused_add_norm.")

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if Mamba is None:
        raise ImportError(
            "OriginalGraphMamba requires mamba_ssm. Activate the environment that has "
            "mamba_ssm installed before using --model_name GraphMambaOriginal."
        )
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    if rms_norm and RMSNorm is None:
        warnings.warn(
            "mamba_ssm RMSNorm is unavailable; falling back to nn.LayerNorm for OriginalGraphMamba.",
            RuntimeWarning,
        )
        rms_norm = False
    if fused_add_norm and (RMSNorm is None or layer_norm_fn is None or rms_norm_fn is None):
        warnings.warn(
            "mamba_ssm fused layernorm kernels are unavailable; disabling fused_add_norm for OriginalGraphMamba.",
            RuntimeWarning,
        )
        fused_add_norm = False
    norm_impl = nn.LayerNorm if not rms_norm else RMSNorm
    norm_cls = partial(norm_impl, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class OriginalGraphMambaClassifier(nn.Module):
    def __init__(
        self,
        band: int,
        num_classes: int,
        patch_size: int,
        depth: int = 6,
        embed_dim: int = 64,
        gcn_layers: int = 3,
        ssm_cfg=None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        device=None,
        dtype=None,
        if_abs_pos_embed: bool = True,
        if_rope: bool = False,
        if_rope_residual: bool = True,
        bimamba_type: str = "v2",
        graph_neighbor_size: int = 3,
        graph_sigma: float = 10.0,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)

        self.residual_in_fp32 = residual_in_fp32
        if rms_norm and RMSNorm is None:
            warnings.warn(
                "mamba_ssm RMSNorm is unavailable; falling back to nn.LayerNorm for OriginalGraphMamba.",
                RuntimeWarning,
            )
            rms_norm = False
        if fused_add_norm and (RMSNorm is None or layer_norm_fn is None or rms_norm_fn is None):
            warnings.warn(
                "mamba_ssm fused layernorm kernels are unavailable; disabling fused_add_norm for OriginalGraphMamba.",
                RuntimeWarning,
            )
            fused_add_norm = False
        self.fused_add_norm = fused_add_norm
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.bimamba_type = bimamba_type
        self.num_tokens = 0
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = patch_size * patch_size
        self.graph_neighbor_size = graph_neighbor_size
        self.graph_sigma = graph_sigma

        self.patch_to_embedding = nn.Linear(band, embed_dim)

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.layer_GCN = nn.Sequential()
        for i in range(depth):
            self.layer_GCN.add_module(
                "GCN_Branch" + str(i),
                OriginalGraphMambaGCN(
                    height=patch_size,
                    width=patch_size,
                    changel=embed_dim,
                    layers_count=gcn_layers,
                ),
            )

        norm_impl = nn.LayerNorm if not rms_norm else RMSNorm
        self.norm_f = norm_impl(embed_dim, eps=norm_epsilon, **factory_kwargs)
        self.pre_logits = nn.Identity()

        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.skipcat = nn.ModuleList()
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(self.num_patches, self.num_patches, [1, 2], 1, 0))

        self.register_buffer(
            "local_graph_mask",
            self._build_local_graph_mask(patch_size, graph_neighbor_size),
            persistent=False,
        )

    @staticmethod
    def _build_local_graph_mask(patch_size: int, neighbor_size: int) -> Tensor:
        if neighbor_size % 2 != 1:
            raise ValueError("graph_neighbor_size must be odd.")
        radius = neighbor_size // 2
        mask = torch.zeros((patch_size * patch_size, patch_size * patch_size), dtype=torch.float32)
        for i in range(patch_size):
            for j in range(patch_size):
                m = i * patch_size + j
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni = i + di
                        nj = j + dj
                        n = ni * patch_size + nj
                        if 0 <= ni < patch_size and 0 <= nj < patch_size and m != n:
                            mask[m, n] = 1.0
        return mask

    def _to_sequence(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            if x.shape[1] == self.patch_to_embedding.in_features:
                x = x.permute(0, 2, 3, 1)
            batch, height, width, channels = x.shape
            if height != self.patch_size or width != self.patch_size:
                raise ValueError(
                    f"Expected patch size {self.patch_size}x{self.patch_size}, got {height}x{width}."
                )
            return x.reshape(batch, height * width, channels)
        if x.dim() == 3:
            if x.shape[1] != self.num_patches:
                raise ValueError(f"Expected {self.num_patches} patch tokens, got {x.shape[1]}.")
            return x
        raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    def build_adjacency(self, x: Tensor) -> Tensor:
        prod = torch.bmm(x, x.transpose(1, 2))
        norm = prod.diagonal(dim1=1, dim2=2).unsqueeze(2)
        dist = (norm + norm.transpose(1, 2) - 2 * prod).clamp(min=0)
        A = torch.exp(-dist / (self.graph_sigma ** 2))
        return A * self.local_graph_mask.to(device=x.device, dtype=x.dtype).unsqueeze(0)

    def center_positions(self, batch_size: int, device: torch.device) -> Tensor:
        center = self.num_patches // 2
        return torch.full((batch_size,), center, dtype=torch.long, device=device)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward_features(self, x: Tensor, batch_A: Tensor, inference_params=None) -> Tensor:
        x = self.patch_to_embedding(x)

        # The official GraphMamba code creates pos_embed when requested but leaves
        # this addition commented out in forward_features; keep that behavior.
        residual = None
        hidden_states = x
        last_output = []

        for layer_idx, (mamba_block, gcn_block) in enumerate(zip(self.layers, self.layer_GCN)):
            last_output.append(hidden_states)
            if layer_idx > 1:
                hidden_states = self.skipcat[layer_idx - 2](
                    torch.cat(
                        [hidden_states.unsqueeze(3), last_output[layer_idx - 2].unsqueeze(3)],
                        dim=3,
                    )
                ).squeeze(3)
            hidden_states, residual = mamba_block(
                hidden_states,
                residual,
                inference_params=inference_params,
            )
            hidden_states = gcn_block(hidden_states, batch_A)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

    def forward(
        self,
        x: Tensor,
        center_pos: Optional[Tensor] = None,
        batch_A: Optional[Tensor] = None,
        return_features: bool = False,
        inference_params=None,
    ) -> Tensor:
        x = self._to_sequence(x).to(torch.float32)
        if batch_A is None:
            batch_A = self.build_adjacency(x)
        else:
            batch_A = batch_A.to(device=x.device, dtype=x.dtype)
        if center_pos is None:
            center_pos = self.center_positions(x.shape[0], x.device)
        else:
            center_pos = center_pos.to(device=x.device, dtype=torch.long)

        x = self.forward_features(x, batch_A, inference_params)
        if return_features:
            return x

        x = self.head(x)
        gather_index = center_pos.view(-1, 1, 1).expand(-1, 1, self.num_classes)
        return x.gather(1, gather_index).squeeze(1)
