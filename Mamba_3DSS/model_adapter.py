import math
import torch
import torch.nn as nn

from .videomamba import VisionMamba


class Mamba3DSSClassifier(nn.Module):
    def __init__(
        self,
        band: int,
        num_classes: int,
        patch_size: int,
        pca_components: int | None = None,
        depth: int = 1,
        embed_dim: int = 32,
        d_state: int = 16,
        group_type: str = "Cube",
        scan_type: str = "Parallel spectral-spatial",
        k_group: int = 4,
        conv3D_channel: int = 32,
        conv3D_kernel: tuple[int, int, int] = (3, 5, 5),
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        fc_drop_rate: float = 0.0,
    ):
        super().__init__()

        # 如果你在 demo 的数据管线已经做了 PCA，则 band 就是 PCA 后的维度
        in_band = band

        dt_rank = math.ceil(embed_dim / 16)
        d_inner = 2 * embed_dim

        # 这些维度在原仓库 config.py 里是这么算的
        dim_patch = patch_size - conv3D_kernel[1] + 1
        dim_linear = in_band - conv3D_kernel[0] + 1

        self.core = VisionMamba(
            group_type=group_type,
            k_group=k_group,
            depth=depth,
            embed_dim=embed_dim,
            dt_rank=dt_rank,
            d_inner=d_inner,
            d_state=d_state,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            fc_drop_rate=fc_drop_rate,
            scan_type=scan_type,
            pos=False,
            cls=False,
            conv3D_channel=conv3D_channel,
            conv3D_kernel=conv3D_kernel,
            dim_patch=dim_patch,
            dim_linear=dim_linear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, band, H, W) -> (B, 1, band, H, W)
        if x.dim() != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        logits, _feature = self.core(x)
        return logits
