import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSWT2D(nn.Module):
    """Single-level 2D SWT implemented by fixed depthwise convolutions (Haar)."""

    def __init__(self):
        super().__init__()
        sqrt2 = math.sqrt(2.0)
        low = torch.tensor([1.0 / sqrt2, 1.0 / sqrt2], dtype=torch.float32)
        high = torch.tensor([-1.0 / sqrt2, 1.0 / sqrt2], dtype=torch.float32)

        ll = torch.outer(low, low)
        lh = torch.outer(low, high)
        hl = torch.outer(high, low)
        hh = torch.outer(high, high)

        # (4, 1, 2, 2): LL, LH, HL, HH
        base_kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("base_kernels", base_kernels, persistent=False)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        kernels = self.base_kernels.repeat(c, 1, 1, 1)  # (4C, 1, 2, 2)

        # padding=1 keeps SWT stationary behavior; crop back to original H,W
        y = F.conv2d(x, kernels, stride=1, padding=1, groups=c)
        y = y[:, :, :h, :w].contiguous()
        y = y.view(b, c, 4, h, w)

        ll = y[:, :, 0, :, :]
        lh = y[:, :, 1, :, :]
        hl = y[:, :, 2, :, :]
        hh = y[:, :, 3, :, :]
        return ll, lh, hl, hh


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DetailCNN(nn.Module):
    def __init__(self, channels: int, depth: int = 3):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(ConvBNAct(channels, channels, k=3, p=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GlobalTransformer(nn.Module):
    def __init__(self, dim: int, depth: int = 2, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        ff_dim = int(dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # (B, HW, C)
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()


class FrequencyGatedFeatureFusion(nn.Module):
    """
    Frequency-Gated Feature Fusion:
      F_c = concat(F_global, F_detail)
      F_m = Conv1x1(F_c)
      G = sigmoid(F_m)
      F = G * F_global + (1 - G) * F_detail

    For channel-mismatch robustness, branch features are first aligned to fusion_dim.
    """

    def __init__(self, global_dim: int, detail_dim: int, fusion_dim: int):
        super().__init__()
        self.global_align = ConvBNAct(global_dim, fusion_dim, k=1, p=0)
        self.detail_align = ConvBNAct(detail_dim, fusion_dim, k=1, p=0)
        self.map_proj = nn.Conv2d(fusion_dim * 2, fusion_dim, kernel_size=1, bias=True)
        self.gate_act = nn.Sigmoid()

    def forward(self, f_global: torch.Tensor, f_detail: torch.Tensor) -> torch.Tensor:
        fg = self.global_align(f_global)
        fd = self.detail_align(f_detail)

        f_c = torch.cat([fg, fd], dim=1)
        f_m = self.map_proj(f_c)
        g = self.gate_act(f_m)
        return g * fg + (1.0 - g) * fd


class ConcatConvFusion(nn.Module):
    """Baseline fusion: align -> concat -> 1x1 conv."""

    def __init__(self, global_dim: int, detail_dim: int, fusion_dim: int):
        super().__init__()
        self.global_align = ConvBNAct(global_dim, fusion_dim, k=1, p=0)
        self.detail_align = ConvBNAct(detail_dim, fusion_dim, k=1, p=0)
        self.fuse_proj = nn.Conv2d(fusion_dim * 2, fusion_dim, kernel_size=1, bias=True)

    def forward(self, f_global: torch.Tensor, f_detail: torch.Tensor) -> torch.Tensor:
        fg = self.global_align(f_global)
        fd = self.detail_align(f_detail)
        return self.fuse_proj(torch.cat([fg, fd], dim=1))


class FFDBNet(nn.Module):
    """
    Frequency-Feature Dual-Branch model with ablation switches:
      - use_swt=True: use Conv-SWT LL/LH/HL/HH decomposition
      - use_swt=False: bypass SWT and feed raw x to both branches
      - fusion_type='gated': FrequencyGatedFeatureFusion (original)
      - fusion_type='concat_conv': concat + 1x1 conv baseline
    """

    def __init__(
        self,
        band: int,
        num_classes: int,
        patch_size: int,
        global_dim: int = 96,
        detail_dim: int = 96,
        fusion_dim: int = 128,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        detail_depth: int = 3,
        dropout: float = 0.0,
        use_swt: bool = True,
        fusion_type: str = "concat_conv",
    ):
        super().__init__()
        self.band = band
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.use_swt = use_swt
        self.fusion_type = fusion_type.lower()

        self.swt = ConvSWT2D()

        self.global_proj = ConvBNAct(band, global_dim, k=1, p=0)
        self.global_branch = GlobalTransformer(
            dim=global_dim,
            depth=transformer_depth,
            num_heads=transformer_heads,
            mlp_ratio=2.0,
            dropout=dropout,
        )

        self.detail_proj = ConvBNAct(band * 3, detail_dim, k=1, p=0)
        self.detail_branch = DetailCNN(channels=detail_dim, depth=detail_depth)

        if self.fusion_type == "gated":
            self.fusion = FrequencyGatedFeatureFusion(
                global_dim=global_dim,
                detail_dim=detail_dim,
                fusion_dim=fusion_dim,
            )
        elif self.fusion_type == "concat_conv":
            self.fusion = ConcatConvFusion(
                global_dim=global_dim,
                detail_dim=detail_dim,
                fusion_dim=fusion_dim,
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        if self.use_swt:
            ll, lh, hl, hh = self.swt(x)
            global_in = ll + (lh + hl + hh) / 3.0
            detail_in = torch.cat([lh, hl, hh], dim=1)
        else:
            global_in = x
            # Keep detail branch input channels unchanged (band * 3).
            detail_in = torch.cat([x, x, x], dim=1)

        global_in = self.global_proj(global_in)
        f_global = self.global_branch(global_in)

        detail_in = self.detail_proj(detail_in)
        f_detail = self.detail_branch(detail_in)

        fused = self.fusion(f_global, f_detail)
        logits = self.classifier(fused)
        return logits
