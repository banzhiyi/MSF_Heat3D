import torch
import torch.nn as nn


class GSSM(nn.Module):
    """
    Gated Spatial-Spectral Merging
    Input : (B, C, P, P)
    Output: (B, C)
    """
    def __init__(self, bands: int, patch_size: int):
        super().__init__()
        # depthwise conv over spatial dims
        self.dw_conv = nn.Conv2d(
            in_channels=bands,
            out_channels=bands,
            kernel_size=patch_size,
            padding=patch_size // 2,
            groups=bands,
            bias=True,
        )
        # pointwise conv to generate gates
        self.pw_conv = nn.Conv2d(
            in_channels=bands,
            out_channels=bands,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.pw_conv(self.dw_conv(x)))
        x = x * gate
        x = x.mean(dim=(-1, -2))  # (B, C)
        return x

class SpectralMambaBlock(nn.Module):
    """
    Token-mixing block on spectral chunks.
    Input/Output: (B, R, D)
    """

    def __init__(self, dim: int, expansion: int = 4, drop: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        x = self.drop(x)
        return x + residual

class SpectralMambaPatch(nn.Module):
    """
    Patchwise SpectralMamba-like baseline for HSI classification.
    Input : (B, C, P, P)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 9,
        chunk_size: int = 8,
        depth: int = 6,
        expansion: int = 4,
        drop: float = 0.0,
    ):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if bands % chunk_size != 0:
            raise ValueError(f"bands ({bands}) must be divisible by chunk_size ({chunk_size})")

        self.bands = bands
        self.chunk_size = chunk_size
        self.ranks = bands // chunk_size

        self.gssm = GSSM(bands=bands, patch_size=patch_size)

        self.blocks = nn.ModuleList(
            [SpectralMambaBlock(dim=chunk_size, expansion=expansion, drop=drop) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(chunk_size)
        self.classifier = nn.Linear(chunk_size, num_classes)

    @staticmethod
    def piecewise_spectral_scan(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """
        x: (B, C)
        return: (B, R, chunk_size)
        """
        b, c = x.shape
        if c % chunk_size != 0:
            raise ValueError(f"channels ({c}) must be divisible by chunk_size ({chunk_size})")
        r = c // chunk_size
        return x.view(b, r, chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, P, P)
        x = self.gssm(x)  # (B, C)
        x = self.piecewise_spectral_scan(x, self.chunk_size)  # (B, R, chunk_size)

        for blk in self.blocks:
            x = blk(x)

        x = x.mean(dim=1)  # (B, chunk_size)
        x = self.norm(x)
        x = self.classifier(x)  # (B, num_classes)
        return x













