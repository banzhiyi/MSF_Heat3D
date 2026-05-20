import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, bottleneck_ratio: int = 2):
        super(ResidualBlock3D, self).__init__()
        mid_channels = max(channels // bottleneck_ratio, 4)

        self.conv1 = nn.Conv3d(
            channels, mid_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(
            mid_channels, channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(channels)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act(out)
        return out


def make_res_stack(channels: int, depth: int, bottleneck_ratio: int = 2) -> nn.Sequential:
    return nn.Sequential(*[
        ResidualBlock3D(channels, bottleneck_ratio=bottleneck_ratio)
        for _ in range(depth)
    ])


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super(SEBlock3D, self).__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class HSI3DCNN(nn.Module):
    """
    计算增强版 3D-CNN HSI 分类模型
    输入:  x [B, band, H, W]
    输出:  logits [B, num_classes]
    """

    def __init__(
        self,
        band: int,
        num_classes: int,
        patch_size: int,
        res2_depth: int = 3,
        res3_depth: int = 3,
    ):
        super(HSI3DCNN, self).__init__()
        self.band = band
        self.num_classes = num_classes
        self.patch_size = patch_size

        # ----- Stage 1: 1 -> 32 -----
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=(7, 3, 3),
            stride=(1, 1, 1),
            padding=(3, 1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(32)
        self.se1 = SEBlock3D(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # ----- Stage 2: 32 -> 64 -----
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 3, 3),
            stride=(1, 1, 1),
            padding=(2, 1, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(64)
        self.se2 = SEBlock3D(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Stage2 残差块放在 pool2 前，保留 3x3 空间网格以提高 3D 卷积计算量。
        self.res_block2 = make_res_stack(64, depth=res2_depth, bottleneck_ratio=2)

        # ----- Stage 3: 64 -> 128 -----
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm3d(128)
        self.se3 = SEBlock3D(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Stage3 增加一层残差块，进一步体现 3D-CNN 在光谱维上的计算开销。
        self.res_block3 = make_res_stack(128, depth=res3_depth, bottleneck_ratio=2)

        # ----- Global pooling & classifier -----
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 先对 128-d 特征做 LayerNorm（更稳定）
        self.feature_norm = nn.LayerNorm(128)

        self.fc1 = nn.Linear(128, 256, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, band, H, W]
        return: logits [B, num_classes]
        """
        B, C, H, W = x.shape
        assert C == self.band, f"输入 band 数 {C} 与初始化时设定的 {self.band} 不一致"

        # [B, band, H, W] -> [B, 1, D=band, H, W]
        x = x.unsqueeze(1)

        # ----- Stage 1 -----
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.silu(x, inplace=True)
        x = self.se1(x)
        x = self.pool1(x)

        # ----- Stage 2 -----
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.silu(x, inplace=True)
        x = self.se2(x)
        x = self.res_block2(x)
        x = self.pool2(x)

        # ----- Stage 3 -----
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.silu(x, inplace=True)
        x = self.se3(x)

        # 视 patch 大小决定是否再池一次，避免 0 尺寸
        if x.size(-1) >= 2 and x.size(-2) >= 2:
            x = self.pool3(x)

        x = self.res_block3(x)

        # ----- Global pooling & classifier -----
        x = self.global_pool(x)       # [B, 128, 1, 1, 1]
        x = x.view(B, -1)             # [B, 128]

        # LayerNorm 在 channel 维度上
        x = self.feature_norm(x)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.silu(x, inplace=True)
        x = self.dropout(x)
        logits = self.classifier(x)   # [B, num_classes]

        return logits
