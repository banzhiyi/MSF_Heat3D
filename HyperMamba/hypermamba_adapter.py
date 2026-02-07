import torch
import torch.nn as nn

from .vmamba import VSSM


class HyperMambaClassifier(nn.Module):
    """
    说明：
    - 适配当前工程：输入 (B, band, patch, patch)，输出 (B, num_classes)
    - 先用一套保守的默认超参把训练/测试流程跑通
    """
    def __init__(
        self,
        band: int,
        num_classes: int,
        patch_size: int,
        depths=(1, 1, 1, 1),
        dims=(64, 128, 256, 512),
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.backbone = VSSM(
            patch_size=patch_size,
            in_chans=band,
            num_classes=num_classes,
            depths=list(depths),
            dims=list(dims),
            # SSM\&MLP 的关键参数给一个“能动起来”的默认值
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v2",
            patchembed_version="v2",
            gmlp=False,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)