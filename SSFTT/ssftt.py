import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


class SSFTTnet(nn.Module):
    def __init__(
        self,
        band: int,
        num_classes: int,
        patch_size: int,
        num_tokens: int = 4,
        dim: int = 64,
        depth: int = 1,
        heads: int = 8,
        mlp_dim: int = 128,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()

        self.band = band
        self.patch_size = patch_size
        self.num_tokens = num_tokens
        self.dim = dim

        # ------------- 3D 卷积特征提取（示例）-------------
        # 注意：这里 C3 和 D 要与你实际代码保持一致
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        # 假设 conv3d 输出通道为 C3=8，输出深度 D = band（或别的值，取决于你的网络）
        C3 = 8
        D = band  # 如果你的 conv3d 改变了深度，这里要按实际输出修改
        in_channels_2d = C3 * D   # rearrange 后 2D 输入通道数

        # ------------- 2D 卷积特征提取：在 __init__ 里直接定义 -------------
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels_2d, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ------------- Transformer / 分类头：保持你原来的实现 -------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # +1 是 cls token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens + 1, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.to_cls_token = nn.Identity()
        self.nn1 = nn.Linear(dim, num_classes)

    def forward(self, x_3d: torch.Tensor):
        # x_3d: [B, 1, band, H, W]
        x = self.conv3d_features(x_3d)                 # [B, C3, D, H, W]
        x_2d = rearrange(x, 'b c d h w -> b (c d) h w')  # [B, C3*D, H, W]

        # 这里直接用在 __init__ 中注册好的 conv2d_features
        x_2d = self.conv2d_features(x_2d)              # [B, 64, H, W]

        B, C, H, W = x_2d.shape
        x_flat = x_2d.flatten(2).transpose(1, 2)       # [B, L, C]

        L = x_flat.size(1)
        if L >= self.num_tokens:
            idx = torch.linspace(0, L - 1, steps=self.num_tokens, device=x_flat.device).long()
            tokens = x_flat.index_select(1, idx)
        else:
            pad = self.num_tokens - L
            pad_tokens = x_flat[:, -1:, :].repeat(1, pad, 1)
            tokens = torch.cat([x_flat, pad_tokens], dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,dim]
        x_tok = torch.cat([cls_tokens, tokens], dim=1) # [B,num_tokens+1,dim]
        x_tok = x_tok + self.pos_embedding
        x_tok = self.dropout(x_tok)

        x_tok = self.transformer(x_tok)
        x_cls = self.to_cls_token(x_tok[:, 0])
        out = self.nn1(x_cls)
        return out

class SSFTT(nn.Module):
    def __init__(
        self,
        band: int,
        num_classes: int,
        patch_size: int,
        num_tokens: int = 4,
        dim: int = 64,
        depth: int = 1,
        heads: int = 8,
        mlp_dim: int = 128,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()
        self.band = band
        self.patch_size = patch_size

        self.backbone = SSFTTnet(
            band=band,
            num_classes=num_classes,
            patch_size=patch_size,
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

    def forward(self, x):
        # x: [B, band, patch, patch]
        assert x.dim() == 4, f'输入 x 应为 [B, band, H, W]，当前形状为 {x.shape}'
        b, c, h, w = x.shape
        assert c == self.band
        assert h == self.patch_size and w == self.patch_size

        x_3d = x.unsqueeze(1)  # [B, 1, band, H, W]
        out = self.backbone(x_3d)
        return out


if __name__ == '__main__':
    model = SSFTTnet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 13, 13)
    y = model(input)
    print(y.size())