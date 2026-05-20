import torch
import torch.nn as nn


class HSI2DCNN(nn.Module):
    """
    HSI 2D-CNN baseline for patch-based classification.

    Compatible with current pipeline:
      - input:  (B, band, patch, patch)
      - output: (B, num_classes)
    """

    def __init__(
        self,
        band: int,
        num_classes: int,
        patch_size: int,
        base_channels: int = 32,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Stem: treat spectral bands as input channels
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            nn.Conv2d(band, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),

            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),

            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
        )

        # Head: global pooling -> vector -> classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(c3, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, band, patch, patch)
        x = self.features(x)
        logits = self.head(x)
        return logits