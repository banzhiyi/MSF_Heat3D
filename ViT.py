import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
    将输入图像 [B, C, H, W] 划分为 patch 并映射到 D 维:
    输出: [B, N, D], 其中 N = (H / patch_size) * (W / patch_size)
    """
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 使用 Conv2d 完成 patch 划分 + 线性映射
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            # 简单 resize 到指定大小（保证 H,W 可被 patch_size 整除）
            x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,  # 输入形状 [B, N, D]
            bias=qkv_bias,
        )
        self.drop_path = nn.Identity()  # 这里不实现 DropPath，保持简洁
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention
        x_residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_residual + attn_out

        # MLP
        x_residual = x
        x_norm = self.norm2(x)
        x = x_residual + self.mlp(x_norm)
        return x

class ViT(nn.Module):
    """
    精简版 Vision Transformer:
    - 输入:  [B, band, H, W]
    - 输出:  [B, num_classes]
    - 接口:  ViT(band, num_classes, img_size, vit_patch_size=4, ...)
    其中 img_size 建议传入 args.patches (例如 7, 9 等 HSI patch 大小)
    """
    def __init__(
        self,
        band: int,
        num_classes: int,
        img_size: int,
        vit_patch_size: int = 1,
        embed_dim: int = 192,     # 类似 ViT-Small/轻量版
        depth: int = 6,           # Transformer 层数
        num_heads: int = 3,       # 注意力头数 (需整除 embed_dim)
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.img_size = img_size

        # patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=vit_patch_size,
            in_chans=band,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # cls token + position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # 初始化 \+ 位置编码
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, band, H, W]
        B = x.shape[0]

        x = self.patch_embed(x)  # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, D]

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_feat = x[:, 0]  # [B, D]
        logits = self.head(cls_feat)  # [B, num_classes]
        return logits


