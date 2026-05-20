import math
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:
    Mamba = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except Exception:
    selective_scan_fn = None
    selective_scan_ref = None


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class FallbackSequenceMixer(nn.Module):
    """
    Pure PyTorch sequence mixer used when mamba_ssm is unavailable or when the
    installed Mamba CUDA kernels cannot run on the current device.
    """

    def __init__(self, dim: int, expand: int = 2, d_conv: int = 5, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * expand)
        padding = d_conv // 2
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=d_conv, padding=padding, groups=dim)
        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x + self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(x) * torch.sigmoid(gate)
        x = self.out_proj(self.drop(x))
        return x + residual


class SafeMambaMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank="auto",
        drop: float = 0.0,
    ):
        super().__init__()
        self.mamba_disabled = Mamba is None
        self.warned_fallback = False
        self.mamba = None
        if Mamba is not None:
            try:
                self.mamba = Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                )
            except Exception as exc:
                self.mamba_disabled = True
                warnings.warn(f"3DSS-Mamba falls back to PyTorch mixer: {type(exc).__name__}: {exc}")
        self.fallback = FallbackSequenceMixer(dim=dim, expand=expand, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_tracing():
            return self.fallback(x)
        if self.mamba is not None and x.is_cuda and not self.mamba_disabled:
            try:
                return self.mamba(x)
            except RuntimeError as exc:
                self.mamba_disabled = True
                if not self.warned_fallback:
                    warnings.warn(
                        "mamba_ssm failed at runtime; using the PyTorch fallback mixer "
                        f"for this 3DSS-Mamba model. Original error: {exc}"
                    )
                    self.warned_fallback = True
        return self.fallback(x)


class SpectralSpatialScan(nn.Module):
    def __init__(self, scan_type: str = "Parallel spectral-spatial"):
        super().__init__()
        self.scan_type = scan_type

    @staticmethod
    def _spectral_priority(x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = x.shape
        return x.permute(0, 2, 3, 1, 4).reshape(b, h * w * t, c)

    @staticmethod
    def _spatial_priority(x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = x.shape
        return x.reshape(b, t * h * w, c)

    @staticmethod
    def _restore_spectral_priority(y: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        b, _, c = y.shape
        return y.reshape(b, h, w, t, c).permute(0, 3, 1, 2, 4).contiguous()

    @staticmethod
    def _restore_spatial_priority(y: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        b, _, c = y.shape
        return y.reshape(b, t, h, w, c).contiguous()

    def make_sequences(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, str, bool]]:
        scan_type = self.scan_type.lower()
        spectral = self._spectral_priority(x)
        spatial = self._spatial_priority(x)

        if scan_type == "spectral-priority":
            return [(spectral, "spectral", False), (torch.flip(spectral, dims=[1]), "spectral", True)]
        if scan_type == "spatial-priority":
            return [(spatial, "spatial", False), (torch.flip(spatial, dims=[1]), "spatial", True)]
        if scan_type == "cross spectral-spatial":
            return [(spectral, "spectral", False), (torch.flip(spatial, dims=[1]), "spatial", True)]
        if scan_type == "cross spatial-spectral":
            return [(spatial, "spatial", False), (torch.flip(spectral, dims=[1]), "spectral", True)]
        if scan_type == "parallel spectral-spatial":
            return [
                (spectral, "spectral", False),
                (torch.flip(spectral, dims=[1]), "spectral", True),
                (spatial, "spatial", False),
                (torch.flip(spatial, dims=[1]), "spatial", True),
            ]

        raise ValueError(f"Unsupported 3DSS scan_type: {self.scan_type}")

    def restore(self, y: torch.Tensor, route: str, reversed_route: bool, t: int, h: int, w: int) -> torch.Tensor:
        if reversed_route:
            y = torch.flip(y, dims=[1])
        if route == "spectral":
            return self._restore_spectral_priority(y, t, h, w)
        if route == "spatial":
            return self._restore_spatial_priority(y, t, h, w)
        raise ValueError(f"Unsupported route: {route}")


class Mamba3DSSBlock(nn.Module):
    """
    3-D spectral-spatial Mamba block.
    Input/Output shape: [B, T, H, W, C]
    """

    def __init__(
        self,
        dim: int,
        d_inner: int,
        d_state: int = 16,
        dt_rank="auto",
        scan_type: str = "Parallel spectral-spatial",
        expand: int = 2,
        drop: float = 0.0,
    ):
        super().__init__()
        self.scan = SpectralSpatialScan(scan_type)
        route_count = len(self.scan.make_sequences(torch.empty(1, 1, 1, 1, dim)))
        self.route_count = route_count
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.force_fp32 = True
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        self.act = nn.SiLU()
        self.conv3d = nn.Conv3d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=1,
            groups=d_inner,
            bias=True,
        )

        self.x_proj = [
            nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
            for _ in range(route_count)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([proj.weight for proj in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            self.dt_init(dt_rank, d_inner)
            for _ in range(route_count)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([proj.weight for proj in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([proj.bias for proj in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(d_state, d_inner, copies=route_count, merge=True)
        self.Ds = self.D_init(d_inner, copies=route_count, merge=True)
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)
        self.dropout = nn.Dropout(drop)

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, merge=True):
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        a_log = torch.log(a).repeat(d_inner, 1)
        if copies > 0:
            a_log = a_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                a_log = a_log.flatten(0, 1)
        a_log = nn.Parameter(a_log)
        a_log._no_weight_decay = True
        return a_log

    @staticmethod
    def D_init(d_inner, copies=1, merge=True):
        d = torch.ones(d_inner)
        if copies > 0:
            d = d.unsqueeze(0).repeat(copies, 1)
            if merge:
                d = d.flatten(0, 1)
        d = nn.Parameter(d)
        d._no_weight_decay = True
        return d

    def _run_selective_scan(self, xs: torch.Tensor) -> torch.Tensor:
        b, k, d, l = xs.shape
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.reshape(b, k * d, l)
        dts = dts.contiguous().reshape(b, k * d, l)
        bs = bs.contiguous()
        cs = cs.contiguous()
        a_logs = self.A_logs.float()
        ds = self.Ds.float()
        dt_bias = self.dt_projs_bias.float().reshape(-1)
        a = -torch.exp(a_logs)

        if self.force_fp32:
            xs = xs.float()
            dts = dts.float()
            bs = bs.float()
            cs = cs.float()

        scan_impl = None
        if xs.is_cuda and selective_scan_fn is not None and not torch.jit.is_tracing():
            scan_impl = selective_scan_fn
        elif selective_scan_ref is not None:
            scan_impl = selective_scan_ref

        if scan_impl is None:
            raise RuntimeError(
                "mamba_ssm selective scan is unavailable. Please install mamba-ssm "
                "or run with an environment that provides selective_scan_ref."
            )

        out = scan_impl(
            xs,
            dts,
            a,
            bs,
            cs,
            ds,
            delta_bias=dt_bias,
            delta_softplus=True,
        )
        return out.reshape(b, k, d, l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        z = self.act(z)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        sequences = self.scan.make_sequences(x)
        xs = torch.stack([seq.transpose(1, 2).contiguous() for seq, _, _ in sequences], dim=1)
        out_y = self._run_selective_scan(xs)

        outputs = []
        for idx, (_, route, reversed_route) in enumerate(sequences):
            y = out_y[:, idx].transpose(1, 2).contiguous()
            outputs.append(self.scan.restore(y, route, reversed_route, t, h, w))

        y = torch.stack(outputs, dim=0).sum(dim=0)
        y = self.out_norm(y)
        y = y * z
        return self.dropout(self.out_proj(y))


class Mamba3DSSClassifier(nn.Module):
    """
    Single-file 3DSS-Mamba adapter for this project.

    The public repository uses input cubes shaped [B, 1, spectral, H, W] after
    PCA. This adapter accepts the project's standard [B, band, H, W] tensors,
    inserts the singleton channel dimension internally, and returns logits only.
    """

    def __init__(
        self,
        band: int,
        num_classes: int,
        patch_size: int,
        depth: int = 1,
        embed_dim: int = 32,
        d_state: int = 16,
        d_inner: int = None,
        dt_rank=None,
        scan_type: str = "Parallel spectral-spatial",
        conv3D_channel: int = 32,
        conv3D_kernel: Tuple[int, int, int] = (3, 5, 5),
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        expand: int = 2,
    ):
        super().__init__()
        if patch_size < conv3D_kernel[1] or patch_size < conv3D_kernel[2]:
            raise ValueError(
                f"patch_size={patch_size} is too small for conv3D_kernel={conv3D_kernel}. "
                "Use a larger --patches value or a smaller kernel."
            )
        if band < conv3D_kernel[0]:
            raise ValueError(f"band={band} is too small for conv3D_kernel={conv3D_kernel}")

        d_inner = int(d_inner or embed_dim * 2)
        dt_rank = dt_rank if dt_rank is not None else max(1, (embed_dim + 15) // 16)

        self.band = band
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.sstg = nn.Sequential(
            nn.Conv3d(1, conv3D_channel, kernel_size=conv3D_kernel, bias=False),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(inplace=True),
        )
        self.embedding_spatial_spectral = nn.Linear(conv3D_channel, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(embed_dim),
                        "block": Mamba3DSSBlock(
                            dim=embed_dim,
                            d_inner=d_inner,
                            d_state=d_state,
                            dt_rank=dt_rank,
                            scan_type=scan_type,
                            expand=expand,
                            drop=drop_rate,
                        ),
                        "drop_path": DropPath(dpr[i]),
                    }
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm3d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B, band, H, W], got shape {tuple(x.shape)}")
        b, c, h, w = x.shape
        if c != self.band:
            raise ValueError(f"Expected {self.band} input bands, got {c}")
        if h != self.patch_size or w != self.patch_size:
            x = F.interpolate(x, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False)

        x = x.unsqueeze(1)  # [B, 1, spectral, H, W]
        x = self.sstg(x)  # [B, C3d, T, H', W']
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.embedding_spatial_spectral(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = x + layer["drop_path"](layer["block"](layer["norm"](x)))

        x = self.norm(x)
        return x.mean(dim=(1, 2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(self.head_drop(features))
