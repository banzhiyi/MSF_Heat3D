import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.nn1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.nn1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.nn2(x)
        x = self.drop(x)
        return x


class MAA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_memory = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, memories):
        x = self.norm(x)
        x_kv = x

        q, k, v = (self.to_q(x), *self.to_kv(x_kv).chunk(2, dim=-1))
        memories = self.to_memory(memories)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        memories = rearrange(memories, 'b n (h d) -> b h n d', h=self.heads)

        k = torch.cat((k, memories), dim=2)
        v = torch.cat((v, memories), dim=2)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MAA(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x, memories):
        for _, (attn, ff) in enumerate(self.layers):
            x = attn(x, memories=memories) + x
            x = ff(x) + x

        return x


class _MassFormerCore(nn.Module):
    """
    核心网络：输入 (B, 1, band, H, W)，输出 (B, num_classes)
    """
    def __init__(
        self,
        *,
        band: int,
        patches: int,
        num_classes: int,
        dim: int = 64,
        depth: int = 2,
        heads: int = 8,
        dim_head: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.2,
        emb_dropout: float = 0.1,
    ):
        super().__init__()

        if patches < 5:
            raise ValueError("patches 过小，conv3d/conv2d 会造成尺寸为负，建议 patches>=5。")

        self.band = band
        self.patches = patches

        # conv3d: (B,1,band,H,W) -> (B,8,band-2,H-2,W-2)
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.ReLU(),
        )

        # conv2d 的 in_channels 取决于 band: 8 * (band-2)
        in_ch_2d = 8 * (band - 2)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_ch_2d, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
        )

        # conv2d 后空间尺寸: (H-2,W-2) 再 -2 => (H-4,W-4)
        h2 = patches - 4
        w2 = patches - 4
        num_tokens = h2 * w2

        # Transformer token 维：conv2d 输出 c=64，所以 token dim 固定 64，与 dim 对齐
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim))
        torch.nn.init.normal_(self.pos_embedding, std=0.001)

        # 若 dim != 64，用线性层把 64 映射到 dim；若 dim == 64 则 Identity
        self.to_dim = nn.Identity() if dim == 64 else nn.Linear(64, dim)

        self.dropout = nn.Dropout(dropout)
        self.drop = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

        # 注意：你原实现用的是 MaxPool2d/AvgPool2d 处理 (B,N,C) 这种 3D 张量会不匹配
        # 这里改为 1D pooling，对 token 维 N 做池化
        self.maxpool = nn.MaxPool1d(kernel_size=min(7, num_tokens), stride=min(5, max(1, num_tokens)))
        self.avgpool = nn.AvgPool1d(kernel_size=min(3, num_tokens), stride=min(3, max(1, num_tokens)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,band,H,W)
        x = self.conv3d(x)                     # (B,8,band-2,H-2,W-2)
        x = x.flatten(1, 2)                    # (B, 8*(band-2), H-2, W-2)
        x = self.conv2d(x)                     # (B,64,H-4,W-4)
        x = rearrange(x, "b c h w -> b (h w) c")  # (B,N,64)
        x = self.to_dim(x)                     # (B,N,dim)

        # pooling over tokens N: need (B,dim,N) for 1d pool
        xt = x.transpose(1, 2)                 # (B,dim,N)
        max_token = self.maxpool(xt).transpose(1, 2)  # (B,Nm,dim)
        avg_token = self.avgpool(xt).transpose(1, 2)  # (B,Na,dim)
        memories = torch.cat((avg_token, max_token), dim=1)  # (B,Na+Nm,dim)
        memories = self.drop(memories)

        cls_token = repeat(self.cls_token, "1 n d -> b n d", b=x.shape[0])
        x = torch.cat((cls_token, x), dim=1)   # (B,1+N,dim)
        x = x + self.pos_embedding             # (B,1+N,dim)
        x = self.dropout(x)

        x = self.transformer(x, memories)      # (B,1+N,dim)
        token = x[:, 0]                         # (B,dim)
        logits = self.mlp_head(token)           # (B,num_classes)
        return logits


class MASSFormer(nn.Module):
    """
    项目适配器：
    - 构造签名：MASSFormer(band, num_classes, patches)
    - 输入： (B, band, patches, patches)
    - 输出： (B, num_classes)
    """
    def __init__(self, band: int, num_classes: int, patches: int,
                 dim: int = 64, depth: int = 2, heads: int = 8, dim_head: int = 8,
                 mlp_dim: int = 512, dropout: float = 0.2, emb_dropout: float = 0.1):
        super().__init__()
        self.core = _MassFormerCore(
            band=band,
            patches=patches,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 项目输入: (B,C,H,W) -> core 输入: (B,1,C,H,W)
        if x.dim() != 4:
            raise ValueError(f"期望输入 (B,C,H,W)，得到 {tuple(x.shape)}")
        x = x.unsqueeze(1)
        return self.core(x)