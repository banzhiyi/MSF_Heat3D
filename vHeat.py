# vHeat.py
# No-downsample version + image-level classification (global pooling)
import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class StemLayer(nn.Module):
    """ Simple stem that preserves spatial resolution (no downsampling) """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        # Keep stride=1 so no downsampling
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_first')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Heat2D(nn.Module):
    """
    Heat operator block using DCT/IDCT-like frequency weighting.
    This implementation dynamically adapts to input HxW; freq_embed is resized/proj'd.
    """

    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        # small spatial kernel replaced by 1x1 projection to reduce memory
        #self.conv = nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0)
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        # project freq_embed -> hidden_dim when sizes differ
        # keep a minimal proj (1->hidden_dim) because freq collapse is used in forward
        self.freq_proj = nn.Linear(1, hidden_dim, bias=True)
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )

    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        # early exit for tiny maps
        if H <= 2 or W <= 2:
            x = self.dwconv(x)
            x = self.out_linear(self.out_norm(x.permute(0, 2, 3, 1).contiguous())).permute(0, 3, 1, 2).contiguous()
            return x

        # ensure freq_embed resized to HxW and projected to hidden_dim per spatial location
        if freq_embed is not None and (freq_embed.shape[0] != H or freq_embed.shape[1] != W):
            # freq_embed: (H0, W0, C_freq) -> (H, W, C_freq)
            freq_resized = F.interpolate(
                freq_embed.permute(2, 0, 1).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)  # (H, W, C_freq)
        else:
            freq_resized = freq_embed

        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())  # B, H, W, 2C
        x, z = x.chunk(chunks=2, dim=-1)  # B, H, W, C

        # prepare DCT/IDCT weight maps for current H,W (cached per module)
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None) is not None and getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]

        # DCT-like transforms with 1D convs (narrow memory intermediate views)
        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)

        if self.infer_mode:
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp)
        else:
            # project freq_resized per-channel to match hidden_dim for exponent shaping
            if freq_resized is None:
                k = self.to_k(torch.zeros((self.hidden_dim,), device=x.device))  # fallback
                weight_exp_local = torch.pow(weight_exp[:, :, None], k)
            else:
                # freq_resized: (H, W, Cf) -> project spatially to hidden_dim
                fr = freq_resized
                if fr.dim() == 3:
                    # collapse channel axis by mean then project (keeps memory low)
                    fr_flat = fr.view(-1, fr.shape[-1]).to(x.device)  # (H*W, Cf)
                    fr_collapse = fr_flat.mean(dim=-1, keepdim=True)  # (H*W,1)
                    proj = self.freq_proj(fr_collapse)  # (H*W, hidden_dim)
                    proj = proj.view(H, W, -1)
                else:
                    proj = self.freq_proj(torch.zeros((H, W, 1), device=x.device))
                k = self.to_k(proj.view(-1, proj.shape[-1]))  # (H*W, hidden_dim)
                k = k.view(H, W, -1)
                weight_exp_local = torch.pow(weight_exp[:, :, None], k)

            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp_local)

        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)

        x = self.out_norm(x)

        x = x * nn.functional.silu(z)
        x = self.out_linear(x)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class HeatBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode = False,
        dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm = True,
        layer_scale = None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.op = Heat2D(res=res, dim=dim, hidden_dim=dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None

        self.infer_mode = infer_mode

        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim),
                                       requires_grad=True)

    def _forward(self, x: torch.Tensor, freq_embed):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.op(self.norm1(x), freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, freq_embed)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), freq_embed))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, freq_embed)
        else:
            return self._forward(input, freq_embed)


class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            if isinstance(module, nn.Module):
                x = module(x, *args, **kwargs)
            else:
                x = module(x)
        x = self[-1](x)
        return x


class AutoBandConv(nn.Module):
    def __init__(self, in_chans, mid_chans=8, k1=5, k2=3, groups=1):
        super().__init__()
        pad1 = (k1 - 1) // 2
        pad2 = (k2 - 1) // 2
        self.band_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=mid_chans, kernel_size=k1, stride=1, padding=pad1, groups=groups),
            nn.BatchNorm1d(mid_chans),
            nn.GELU(),
            nn.Conv1d(in_channels=mid_chans, out_channels=1, kernel_size=k2, stride=1, padding=pad2, groups=1),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
        x = self.band_conv(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


class vHeat(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768], drop_path_rate=0.2, patch_norm=True, post_norm=True,
                 layer_scale=None, use_checkpoint=False, mlp_ratio=4.0, img_size=224,
                 act_layer='GELU', infer_mode=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.depths = depths

        # Stem: no downsampling
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=self.embed_dim,
                                     act_layer='GELU',
                                     norm_layer='LN')

        # keep full resolution across stages
        self.res = [img_size] * self.num_layers

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.infer_mode = infer_mode

        # band conv front-end
        self.band_conv = AutoBandConv(in_chans=in_chans, mid_chans=8, k1=5, k2=3)

        # frequency embeddings (init at img_size but will be resized at run-time)
        self.freq_embed = nn.ParameterList()
        for i in range(self.num_layers):
            if i == 0:
                input_channels = self.embed_dim
            else:
                input_channels = self.dims[i - 1]
            self.freq_embed.append(nn.Parameter(torch.zeros(self.res[i], self.res[i], input_channels), requires_grad=True))
            trunc_normal_(self.freq_embed[i], std=.02)

        # build layers: NO spatial downsampling; use 1x1 conv projection when channels change
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                input_channels = self.embed_dim
            else:
                input_channels = self.dims[i_layer - 1]
            output_channels = self.dims[i_layer]

            if input_channels != output_channels:
                proj = nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, bias=False),
                    LayerNorm2d(output_channels)
                )
            else:
                proj = nn.Identity()

            self.layers.append(self.make_layer(
                res=self.res[i_layer],
                dim=input_channels,
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=LayerNorm2d,
                post_norm=post_norm,
                layer_scale=layer_scale,
                downsample=proj,
                mlp_ratio=mlp_ratio,
                infer_mode=infer_mode,
            ))

        # === IMAGE-LEVEL classification head (GLOBAL POOL + Linear) ===
        # Use LayerNorm2d followed by global avg pool and a Linear classifier
        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes)
        )

        self.apply(self._init_weights)

    @staticmethod
    def make_layer(
        res=14,
        dim=96,
        depth=2,
        drop_path=[0.1, 0.1],
        use_checkpoint=False,
        norm_layer=LayerNorm2d,
        post_norm=True,
        layer_scale=None,
        downsample=nn.Identity(),
        mlp_ratio=4.0,
        infer_mode=False,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(HeatBlock(
                res=res,
                dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                post_norm=post_norm,
                layer_scale=layer_scale,
                infer_mode=infer_mode,
            ))

        return AdditionalInputSequential(*blocks, downsample)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def infer_init(self):
        for i, layer in enumerate(self.layers):
            for block in layer[:-1]:
                block.op.infer_init_heat2d(self.freq_embed[i])
        del self.freq_embed

    def forward_features(self, x):
        # band convolution front-end (spectral modeling)
        x = self.band_conv(x)
        x = self.patch_embed(x)

        if self.infer_mode:
            for i, layer in enumerate(self.layers):
                x = layer(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x, self.freq_embed[i])

        return x

    def forward(self, x):
        x = self.forward_features(x)  # (B, C, H, W)
        x = self.classifier(x)        # -> (B, num_classes)
        return x


class S2VHeat(nn.Module):
    """vHeat的S2VNet兼容版本"""

    def __init__(self, band, num_classes, patch_size):
        super().__init__()
        self.backbone = vHeat(
            in_chans=band,
            num_classes=num_classes,
            img_size=patch_size,
            depths=[2, 2, 4, 2],  # 增加中间层深度
            dims=[32, 64, 128, 256],
            mlp_ratio=2.0,
            drop_path_rate=0.1,
            layer_scale=1e-6,  # 添加layer scale
            use_checkpoint=False  # 训练时关闭checkpoint以加速
        )

    def forward(self, x, output_abu=False):
        logits = self.backbone(x)

        # 🆕 只返回分类logits，不返回其他dummy变量
        return logits


if __name__ == "__main__":
    from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis
    model = vHeat(in_chans=200, img_size=27, dims=[128, 128, 128, 128], depths=[2,2,2,2]).cuda()
    input = torch.randn((1, 200, 27, 27), device=torch.device('cuda'))
    analyze = FlopCountAnalysis(model, (input,))
    print(flop_count_str(analyze))
