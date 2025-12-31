import torch
import torch.nn as nn

class HybridSN(nn.Module):
    r"""
    适配当前项目框架的 HybridSN 模型。

    输入:  x, 形状为 (B, band, H, W)，H == W == patches
    输出:  logits, 形状为 (B, num_classes)
    """
    def __init__(self, band: int, num_classes: int, patches: int):
        super().__init__()
        # 与原实现中的 in_chs / patch_size / class_nums 对应
        self.in_chs = band          # 光谱通道数
        self.patch_size = patches   # 空间 patch 尺寸
        self.num_classes = num_classes

        # ========== 3D 卷积部分 ==========
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3)),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True)
        )

        # 使用 dummy 输入推断 3D 卷积输出形状: (1, C3, S', H', W')
        self.x1_shape = self._get_shape_after_3dconv()

        # ========== 2D 卷积部分 ==========
        in_ch_2d = self.x1_shape[1] * self.x1_shape[2]  # C3 * S'
        # 关键修改: 增加 padding=1，避免 1x1 输入时报 kernel size 过大
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch_2d, out_channels=64,
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        # 再用 dummy 输入推断 2D conv 后展平维度
        self.x2_dim = self._get_shape_after_2dconv()

        # ========== 全连接分类头 ==========
        self.dense1 = nn.Sequential(
            nn.Linear(self.x2_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )

        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )

        self.dense3 = nn.Linear(128, self.num_classes)

    def _get_shape_after_3dconv(self):
        r"""
        使用零张量推断 3D 卷积栈输出尺寸。

        输入假设为 (B=1, C=1, S=in_chs, H=patch_size, W=patch_size)
        """
        x = torch.zeros((1, 1, self.in_chs, self.patch_size, self.patch_size))
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        # 返回完整 shape: (1, C3, S', H', W')
        return x.shape

    def _get_shape_after_2dconv(self):
        r"""
        基于 3D 卷积输出，再通过 conv4 推断展平后的维度大小。
        """
        _, C3, S_red, H3, W3 = self.x1_shape   # (1, C3, S', H', W')
        x = torch.zeros((1, C3 * S_red, H3, W3))
        with torch.no_grad():
            x = self.conv4(x)
        # 展平维度 = C_out * H_out * W_out
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        前向传播。

        参数:
            x: 张量, 形状 (B, band, H, W)

        返回:
            logits: 张量, 形状 (B, num_classes)
        """
        B, C, H, W = x.shape

        # 基本的尺寸检查
        if H != self.patch_size or W != self.patch_size:
            raise ValueError(
                f"HybridSN 期望输入空间尺寸为 ({self.patch_size}, {self.patch_size})，"
                f"但当前为 ({H}, {W})"
            )
        if C != self.in_chs:
            raise ValueError(
                f"HybridSN 期望输入光谱通道数为 {self.in_chs}，"
                f"但当前为 {C}"
            )

        # (B, band, H, W) -> (B, 1, band, H, W)，在光谱维上做 3D 卷积
        x = x.unsqueeze(1)

        # 3D 卷积堆叠
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # (B, C3, S', H', W')

        # reshape 成 2D 卷积输入: (B, C3 * S', H', W')
        B, C3, S_red, H3, W3 = x.shape
        x = x.view(B, C3 * S_red, H3, W3)

        # 2D 卷积
        x = self.conv4(x)

        # 展平
        x = x.contiguous().view(B, -1)

        # 全连接分类头
        x = self.dense1(x)
        x = self.dense2(x)
        logits = self.dense3(x)

        return logits