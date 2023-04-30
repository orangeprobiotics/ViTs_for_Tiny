import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch
from functools import partial
from torch.nn import CrossEntropyLoss
from sklearn.decomposition import PCA
from encoder import ConvLayer
from sklearn.cluster import k_means
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.vision_transformer import Block as TimmBlock
from attn1 import FullAttention, ProbAttention
import math
from drloc import DenseRelativeLoc
from SPT import ShiftedPatchTokenization
from fca import MultiSpectralAttentionLayer
from fca import DCTGap


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super(Mlp, self).__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.encoder = nn.Sequential(
#             nn.Linear(in_features, hidden_features),
#             act_layer(),
#             nn.Linear(hidden_features, hidden_features//2),
#             act_layer(),
#             nn.Linear(hidden_features//2, hidden_features//4),
#             act_layer(),
#             nn.Linear(hidden_features//4, 2),
#             nn.Dropout(drop),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(2, hidden_features//4),
#             act_layer(),
#             nn.Linear(hidden_features//4, hidden_features//2),
#             act_layer(),
#             nn.Linear(hidden_features//2, hidden_features),
#             act_layer(),
#             nn.Linear(hidden_features, out_features),
#             nn.Tanh(),
#             nn.Dropout(drop),
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


class SEBlock(nn.Module):
    def __init__(self, mode, channels, ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape  # x为B,C,H,W
        v = self.global_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v


class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use
        # in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size=(k, k), stride=(1, 1), padding=((k - 1) // 2, (k - 1) // 2),
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim=1, keepdim=True)
        max_x, _ = self.max_pooling(x, dim=1, keepdim=True)
        v = self.conv(torch.cat((max_x, avg_x), dim=1))
        v = self.sigmoid(v)
        return x * v

# class CA_Block(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(CA_Block, self).__init__()
#
#         self.fc_layers = nn.Sequential(nn.Linear(in_features=channels, out_features=channels // reduction, bias=False),
#                                        nn.ReLU(),)
#
#         self.F_h = nn.Linear(in_features=channels // reduction, out_features=channels, bias=False)
#         self.F_w = nn.Linear(in_features=channels // reduction, out_features=channels, bias=False)
#
#         self.sigmoid_h = nn.Sigmoid()
#         self.sigmoid_w = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_h = nn.AdaptiveMaxPool2d((h, 1))(x).permute(0, 1, 3, 2)
#         x_w = nn.AdaptiveAvgPool2d((1, w))(x)
#         v = torch.cat((x_h, x_w), 3).permute(0, 2, 3, 1)
#         x_cat_fc_relu = self.fc_layers(v)
#
#         x_cat_fc_split_h, x_cat_fc_split_w = x_cat_fc_relu.split([h, w], 2)
#
#         s_h = self.sigmoid_h(self.F_h(x_cat_fc_split_h)).permute(0, 3, 2, 1)
#         s_w = self.sigmoid_w(self.F_w(x_cat_fc_split_w)).permute(0, 3, 1, 2)
#
#         out = x * s_h.expand_as(x) * s_w.expand_as(x)
#
#         return out


# class SEBlock(nn.Module):
#     def __init__(self, mode, channels, ratio=16):
#         super(SEBlock, self).__init__()
#         self.avg_pooling = nn.AdaptiveAvgPool2d(1)
#         self.max_pooling = nn.AdaptiveMaxPool2d(1)
#         if mode == "max":
#             self.global_pooling = self.max_pooling
#         elif mode == "avg":
#             self.global_pooling = self.avg_pooling
#         d = max(ratio, channels // ratio)
#         self.fc_layers = nn.Sequential(
#             nn.Linear(in_features=channels, out_features=d, bias=False),
#             nn.ReLU(),
#             nn.Linear(in_features=d, out_features=channels, bias=False),
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, _, _ = x.shape  # x为B,C,H,W
#         v = self.global_pooling(x).view(b, c)
#         v = self.fc_layers(v).view(b, c, 1, 1)
#         v = self.sigmoid(v)
#         return x * v


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupAttention(nn.Module):
    """
    LSA: self attention within a group
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """

    def __init__(self, inner_attention, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.inner_attention = inner_attention
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.q = DepthWiseConv1d(dim_in=dim, dim_out=dim, kernel_size=3, padding=1, stride=1, bias=qkv_bias)
        # self.kv = DepthWiseConv1d(dim_in=dim, dim_out=dim * 2, kernel_size=3, padding=1, stride=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear = linear

        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.srl = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # queries = self.q(x.permute(0, 2, 1)).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads)
        queries = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        # q=B,3136, 1, 64
        size1 = H // self.sr_ratio
        size2 = W // self.sr_ratio

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = nn.AdaptiveMaxPool2d((size1, size2))(x_)
            x_ = self.srl(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        keys, values = kv[0], kv[1]
        x, attn = self.inner_attention(
            queries,
            keys,
            values,
        )
        x = x.reshape(B, N, C)  # B, 3136, 64
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# class Attention(nn.Module):
#     """
#     GSA: using a  key to summarize the information for a group to be efficient.
#     """
#
#     def __init__(self, inner_attention, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
#                  sr_ratio=1, linear=False):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#         self.inner_attention = inner_attention
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q = DepthWiseConv2d(dim_in=dim, dim_out=dim, kernel_size=3, padding=1, stride=1, bias=qkv_bias)
#         self.kv = DepthWiseConv2d(dim_in=dim, dim_out=dim * 2, kernel_size=3, padding=1, stride=1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.linear = linear
#
#         self.sr_ratio = sr_ratio
#         if not linear:
#             if sr_ratio > 1:
#                 self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#                 self.norm = nn.LayerNorm(dim)
#         else:
#             self.srl = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
#             self.norm = nn.LayerNorm(dim)
#             self.act = nn.GELU()
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.permute(0, 2, 1).reshape(B, C, H, W)
#         queries = self.q(x).reshape(B, C, -1).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads)
#         size1 = H // self.sr_ratio
#         size2 = W // self.sr_ratio
#
#         if not self.linear:
#             if self.sr_ratio > 1:
#                 x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
#                 x = self.norm(x)
#                 x = x.permute(0, 2, 1).reshape(B, C, size1, size2)
#                 kv = self.kv(x).reshape(B, C, -1).permute(0, 2, 1).\
#                     reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
#             else:
#                 kv = self.kv(x).reshape(B, C, -1).permute(0, 2, 1). \
#                     reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
#         else:
#             x_ = x
#             x_ = nn.AdaptiveMaxPool2d((size1, size2))(x_)
#             x_ = self.srl(x_)
#             x_ = self.norm(x_)
#             x_ = self.act(x_)
#             kv = self.kv(x_).reshape(B, C, -1).permute(0, 2, 1). \
#                 reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
#         keys, values = kv[0], kv[1]
#
#         x, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#         )
#         x = x.reshape(B, N, C)
#         # attn = (q @ k.transpose(-2, -1)) * self.scale
#         # attn = attn.softmax(dim=-1)
#         # attn = self.attn_drop(attn)
#         #
#         # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            # 对每个通道做分组卷积，groups=dim_in
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.BatchNorm2d(dim_in),
            # 实现点乘操作，即所有通道融合并升维
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class DepthWiseConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias, padding_mode='circular'),
            nn.BatchNorm1d(dim_in),
            nn.ELU(),
            nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, attention, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            inner_attention=attention,
            dim=dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(SBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                     drop_path, act_layer)

    def forward(self, x, H, W):
        return super(SBlock, self).forward(x)


class GroupBlock(nn.Module):
    def __init__(self, attention, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if ws == 1:
            self.attn = Attention(
                inner_attention=attention,
                dim=dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, p=2):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=p)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(_init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, (H, W)

class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, factors=None, in_chans=3, num_classes=1000, embed_dims=None,
                 num_heads=None, mlp_ratios=None, qkv_bias=False, qk_scale=None, drop_rate=0., pca_dims=None,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=None, wss=None,
                 depths=None, sr_ratios=None, block_cls=GroupBlock, alpha=0.3, sparse=False, use_drloc=True,
                 drloc_mode="l1"
                 , sample_size=32, use_abs=False, mixup_fun=None, ):
        super().__init__()
        if factors is None:
            factors = [100, 32, 10, 4]
        if wss is None:
            wss = [7, 7, 7, 7]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.mixup_fun = mixup_fun
        self.alpha = alpha
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if depths is None:
            depths = [3, 4, 6, 3]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        # if pca_dims is None:
        #     pca_dims = [72, 144, 288, 576]
        # self.pos_block = nn.ModuleList()
        # self.ppm_block = nn.ModuleList(
        #     [PyramidPooling(embed_dim, embed_dim) for embed_dim in embed_dims]
        # )
        self.num_classes = num_classes
        self.depths = depths
        self.pos_block = nn.ModuleList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims]
        )
        # self.pca_block = nn.ModuleList(
        #     [PyramidPCA(embed_dim) for embed_dim in embed_dims]
        # )
        # for i in range(len(depths)):
        #     self.pos_block.append(PosCNN(in_chans=embed_dims[i], embed_dim=embed_dims[i], depth=i))
        # self.conv_block = nn.ModuleList(
        #     [ConvLayer(embed_dim) for embed_dim in embed_dims]
        # )
        # patch_embed
        self.patch_embeds = nn.ModuleList()
        # self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.use_drloc = use_drloc
        if sparse:
            Attn = ProbAttention
        else:
            Attn = FullAttention

        for i in range(len(depths)):
            if i == 0:
                # self.patch_embeds.append(PatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
                # self.patch_embeds.append(ShiftedPatchTokenization(in_dim=in_chans, dim=embed_dims[i],
                #                                                   patch_size=patch_size))
                self.patch_embeds.append(
                    OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                      embed_dim=embed_dims[i], p=2))

            else:
                # self.patch_embeds.append(
                #     OverlapPatchEmbed(img_size=img_size // (2 ** (i + 1)), patch_size=3, stride=2,
                #                       in_chans=embed_dims[i - 1],
                #                       embed_dim=embed_dims[i], p=1))
                self.patch_embeds.append(
                    ShiftedPatchTokenization(in_dim=embed_dims[i - 1], dim=embed_dims[i], patch_size=2))
                # self.patch_embeds.append(
                #     PatchEmbed(img_size // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            # _block = nn.ModuleList([block_cls(
            #     attention=Attn(scale=qk_scale, factor=factors[k], attention_dropout=attn_drop_rate),
            #     dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
            #     qk_scale=qk_scale,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            #     sr_ratio=sr_ratios[k])
            #     for i in range(depths[k])])
            _block = nn.ModuleList([block_cls(
                attention=Attn(scale=qk_scale, factor=factors[k], attention_dropout=attn_drop_rate),
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k], ws=1 if i % 2 == 1 else wss[k])
                for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        # self.norm = norm_layer(embed_dims[-1])
        # self.bn = nn.BatchNorm2d(embed_dims[-1])
        self.to_latent = nn.Identity()
        # self.sa = Spatial_Attention_Module(k=7)
        c2wh = dict([(64, 56), (128, 28), (320, 14), (512, 7)])
        self.dg = DCTGap(embed_dims[-1], c2wh[embed_dims[-1]], c2wh[embed_dims[-1]], freq_sel_method='top1')
        self.ln = nn.LayerNorm(normalized_shape=[embed_dims[-1], c2wh[embed_dims[-1]], c2wh[embed_dims[-1]]], eps=1e-6)

        # classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        if self.use_drloc:
            self.drloc = DenseRelativeLoc(
                in_dim=embed_dims[-1],
                out_dim=2 if drloc_mode == "l1" else 14,
                sample_size=sample_size,
                drloc_mode=drloc_mode,
                use_abs=use_abs
            )

        # init weights
        # for pos_emb in self.pos_embeds:
        #     trunc_normal_(pos_emb, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for k in range(len(self.depths)):
            for i in range(self.depths[k]):
                self.blocks[k][i].drop_path.drop_prob = dpr[cur + i]
            cur += self.depths[k]

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            # x = self.pca_block[i](x)
            # x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            # for blk in self.blocks[i]:
            #     x = blk(x, H, W)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)
                # else:
                #     if j == len(self.blocks[i]) - 1:
                #         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                #         x = self.ppm_block[i](x)
            # if i < len(self.depths) - 1:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = x.flatten(2).transpose(1, 2)
        # x = self.norm(x)
        # x = self.se(x)
        outs = Munch()
        if self.use_drloc:
            x_last = x
            B, C, H, W = x_last.size()

            drloc_feats, deltaxy = self.drloc(x_last)
            outs.drloc = [drloc_feats]
            outs.deltaxy = [deltaxy]
            outs.plz = [H]  # plane size
        # x = self.sa(x)
        # B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1, 2)
        x = self.ln(x)
        # x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dg(x)

        return self.to_latent(x), outs

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.head(x)
    #
    #     return x
    def forward(self, x, labels=None):
        x1, outs = self.forward_features(x)
        logits = self.head(x1)
        outs.logits = logits
        if labels is not None:
            if self.mixup_fun is not None:
                loss_fct = SoftTargetCrossEntropy()
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            x2, _ = self.forward_features(x)
            newlogits = self.head(x2)
            loss1 = loss_fct(newlogits, labels)
            loss += loss1
            p = torch.log_softmax(logits.view(-1, self.num_classes), dim=-1)
            p_tec = torch.softmax(logits.view(-1, self.num_classes), dim=-1)
            q = torch.log_softmax(newlogits.view(-1, self.num_classes), dim=-1)
            q_tec = torch.softmax(newlogits.view(-1, self.num_classes), dim=-1)
            kl_loss = F.kl_div(p, q_tec, reduction='none').sum()
            reverse_kl_loss = F.kl_div(q, p_tec, reduction='none').sum()

            loss += self.alpha * (kl_loss + reverse_kl_loss)

            return loss
        else:
            return outs




def pvt_tiny(num_classes: int = 1000, sparse=False, use_drloc=True, drloc_mode="l1", sample_size=32, use_abs=False,
             mixup_fun=None):
    model = PyramidVisionTransformer(
        patch_size=4,
        wss=[7, 4, 2, 1],
        factors=[100, 32, 10, 4],
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        num_classes=num_classes,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        sparse=sparse,
        use_drloc=use_drloc,
        drloc_mode=drloc_mode,
        sample_size=sample_size,
        use_abs=use_abs,
        mixup_fun=mixup_fun
    )
    return model


def pvt_small(num_classes: int = 1000):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        num_classes=num_classes,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    )
    return model


# PEG  from https://arxiv.org/abs/2102.10882
# class PosCNN(nn.Module):
#     def __init__(self, in_chans, embed_dim=768, s=1):
#         super(PosCNN, self).__init__()
#         self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
#         self.s = s  # kernel_size为3，stride为s，
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         feat_token = x
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
#         if self.s == 1:
#             x = self.proj(cnn_feat) + cnn_feat
#         else:
#             x = self.proj(cnn_feat)
#         x = x.flatten(2).transpose(1, 2)
#         return x
#
#     def no_weight_decay(self):
#         return ['proj.%d.weight' % i for i in range(4)]

# class PyramidPCA(nn.Module):
#     def __init__(self, out_chans=768):
#         super(PyramidPCA, self).__init__()
#         self.pca = PCA(n_components=out_chans)
#
#     def forward(self, x,):
#         B, N, C = x.shape
#         x = x.reshape(B*N, C)
#         x = x.data.cpu().numpy()
#         self.pca.fit(x)
#         x = self.pca.transform(x)
#         x = torch.tensor(x)
#         x = x.reshape(B, N, -1)
#
#         return x


# class PosCNN(nn.Module):
#     def __init__(self, in_chans, embed_dim=768, s=1):
#         super(PosCNN, self).__init__()
#         inter_channels = int(in_chans / 4)
#         # self.proj = nn.Sequential(nn.Conv2d(in_chans, inter_channels, 1, s, 1, bias=True, groups=inter_channels),
#         # nn.Conv2d(inter_channels, inter_channels, 3, s, 1, bias=True, groups=inter_channels), nn.Conv2d(
#         # inter_channels, embed_dim, 1, s, 1, bias=True, groups=inter_channels), nn.BatchNorm2d(embed_dim), nn.ELU(),
#         # nn.MaxPool2d((3, 3), stride=2, padding=1))
#         self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim),
#                                   nn.BatchNorm2d(in_chans),
#                                   nn.ELU(),
#                                   nn.MaxPool2d((3, 3), stride=2, padding=1))
#         self.s = s  # kernel_size为3，stride为s，
#         # self.se = SEBlock(mode="avg", channels=embed_dim)
#         c2wh = dict([(64, 56), (128, 28), (320, 14), (512, 7)])
#         self.att = MultiSpectralAttentionLayer(embed_dim, c2wh[embed_dim], c2wh[embed_dim], freq_sel_method='top2')
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         feat_token = x
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
#         if self.s == 1:
#             x = self.proj(cnn_feat)
#             x = F.interpolate(x, (H, W), None, "bilinear", align_corners=True)
#             x = x + cnn_feat
#         else:
#             x = self.proj(cnn_feat)
#             x = F.interpolate(x, (H, W), None, "bilinear", align_corners=True)
#         # x = self.se(x)
#         x = self.att(x)
#         x = x.flatten(2).transpose(1, 2)
#         return x
#
#     def no_weight_decay(self):
#         return ['proj.%d.weight' % i for i in range(4)]

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        inter_channels = int(in_chans / 4)
        # self.proj = nn.Sequential(nn.Conv2d(in_chans, inter_channels, 1, s, 1, bias=True, groups=inter_channels),
        # nn.Conv2d(inter_channels, inter_channels, 3, s, 1, bias=True, groups=inter_channels), nn.Conv2d(
        # inter_channels, embed_dim, 1, s, 1, bias=True, groups=inter_channels), nn.BatchNorm2d(embed_dim), nn.ELU(),
        # nn.MaxPool2d((3, 3), stride=2, padding=1))
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim),
                                  nn.BatchNorm2d(in_chans),
                                  nn.ELU(),
                                  nn.MaxPool2d((3, 3), stride=2, padding=1))
        self.s = s
        self.se = SEBlock(mode="avg", channels=embed_dim)
        self.sa = Spatial_Attention_Module(k=7)

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat)
            x = F.interpolate(x, (H, W), None, "bilinear", align_corners=True)
            x = x + cnn_feat
        else:
            x = self.proj(cnn_feat)
            x = F.interpolate(x, (H, W), None, "bilinear", align_corners=True)
        x = self.sa(x)
        x = self.se(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


# class PosCNN(nn.Module):
#     def __init__(self, in_chans, embed_dim=768, s=1, depth=1, ):
#         super(PosCNN, self).__init__()
#         # 输入是四维
#         # inter_channels = int(in_chans / 4)
#         # self.proj = nn.Sequential(nn.Conv2d(in_chans, inter_channels, 1, s, 1, bias=True, groups=inter_channels),
#         # nn.Conv2d(inter_channels, inter_channels, 3, s, 1, bias=True, groups=inter_channels), nn.Conv2d(
#         # inter_channels, embed_dim, 1, s, 1, bias=True, groups=inter_channels), nn.BatchNorm2d(embed_dim), nn.ELU(),
#         # nn.MaxPool2d((3, 3), stride=2, padding=1))
#         self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim),
#                                   nn.BatchNorm2d(in_chans),
#                                   nn.ELU(),
#                                   nn.MaxPool2d((depth*2+1, depth*2+1), stride=2, padding=depth))
#         self.s = s  # kernel_size为3，stride为s，
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         feat_token = x
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)  # 将B,N,C重新变为B*C*H*W 才能用2维卷积
#         if self.s == 1:
#             x = self.proj(cnn_feat)
#             x = F.interpolate(x, (H, W), None, "bilinear", align_corners=True)
#             x = x + cnn_feat
#         else:
#             x = self.proj(cnn_feat)
#             x = F.interpolate(x, (H, W), None, "bilinear", align_corners=True)
#         x = x.flatten(2).transpose(1, 2)
#         return x
#
#     def no_weight_decay(self):
#         return ['proj.%d.weight' % i for i in range(4)]


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, ):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=True, groups=inter_channels),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ELU(),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=True, groups=inter_channels),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ELU(),
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=True, groups=inter_channels),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ELU(),
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=True, groups=inter_channels),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ELU(),
                                   )
        # self.conv2 = nn.Conv2d(in_channels, inter_channels, 1, bias=True, groups=inter_channels)
        # self.conv3 = nn.Conv2d(in_channels, inter_channels, 1, bias=True, groups=inter_channels)
        # self.conv4 = nn.Conv2d(in_channels, inter_channels, 1, bias=True, groups=inter_channels)
        self.pool1 = nn.MaxPool2d((1, 1), stride=2, padding=0)
        self.pool2 = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.pool3 = nn.MaxPool2d((5, 5), stride=2, padding=2)
        self.pool4 = nn.MaxPool2d((7, 7), stride=2, padding=3)
        self.out = nn.Conv2d(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveMaxPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool1(x)), size)
        feat2 = self.upsample(self.conv2(self.pool2(x)), size)
        feat3 = self.upsample(self.conv3(self.pool3(x)), size)
        feat4 = self.upsample(self.conv4(self.pool4(x)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)  # concat 四个池化的结果
        x = self.out(x)
        return x


# class PosCNN(nn.Module):
#     def __init__(self, in_chans, embed_dim=768, s=1):
#         super(PosCNN, self).__init__()
#         self.ppm = PyramidPooling(in_chans, embed_dim)
#         self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim),
#                                   nn.BatchNorm2d(in_chans),
#                                   nn.ELU(),
#                                   nn.MaxPool2d((3, 3), stride=2, padding=1))
#         self.s = s  # kernel_size为3，stride为s，
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         feat_token = x
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
#         x = self.ppm(cnn_feat)
#         x = x + cnn_feat
#         # if self.s == 1:
#         #     x = self.proj(cnn_feat)
#         #     x = F.interpolate(x, (H, W), None, "bilinear", align_corners=True)
#         #     x = x + cnn_feat
#         # else:
#         #     x = self.proj(cnn_feat)
#         #     x = F.interpolate(x, (H, W), None, "bilinear", align_corners=True)
#         x = x.flatten(2).transpose(1, 2)
#         return x
#
#     def no_weight_decay(self):
#         return ['proj.%d.weight' % i for i in range(4)]


# class PosCNN(nn.Module):
#     def __init__(self, in_chans, embed_dim=768, s=1):
#         super(PosCNN, self).__init__()
#         self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim),
#                                   nn.BatchNorm2d(in_chans),
#                                   nn.ELU(), )
#         self.maxpool = nn.MaxPool2d((3, 3), stride=2, padding=1, return_indices=True)
#         self.unpool = nn.MaxUnpool2d((3, 3), stride=2, padding=1)
#         self.s = s  # kernel_size为3，stride为s，
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         feat_token = x
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
#         shape = (B, C, H, W)
#         if self.s == 1:
#             x = self.proj(cnn_feat)
#             x, indices = self.maxpool(x)
#             x = self.unpool(x, indices, output_size=shape)
#             x = x + cnn_feat
#         else:
#             x = self.proj(cnn_feat)
#             x, indices = self.maxpool(x)
#             x = self.unpool(x, indices, output_size=shape)
#         x = x.flatten(2).transpose(1, 2)
#         return x
#
#     def no_weight_decay(self):
#         return ['proj.%d.weight' % i for i in range(4)]


class CPVTV2(PyramidVisionTransformer):
    """
    Use useful results from CPVT. PEG and GAP.
    Therefore, cls token is no longer required.
    PEG is used to encode the absolute position on the fly, which greatly affects the performance when input resolution
    changes during the training (such as segmentation, detection)
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None,
                 num_heads=None, mlp_ratios=None, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 depths=None, sr_ratios=None, block_cls=Block):
        super(CPVTV2, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads, mlp_ratios,
                                     qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, depths,
                                     sr_ratios, block_cls)

        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if depths is None:
            depths = [3, 4, 6, 3]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        del self.pos_embeds
        del self.cls_token
        self.pos_block = nn.ModuleList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def no_weight_decay(self):
        return set(['cls_token'] + ['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x.mean(dim=1)  # GAP here


class PCPVT(CPVTV2):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None,
                 num_heads=None, mlp_ratios=None, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=None, sr_ratios=None, block_cls=SBlock):
        super(PCPVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                    mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                    norm_layer, depths, sr_ratios, block_cls)
        if depths is None:
            depths = [4, 4, 4]
        if sr_ratios is None:
            sr_ratios = [4, 2, 1]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4]
        if num_heads is None:
            num_heads = [1, 2, 4]
        if embed_dims is None:
            embed_dims = [64, 128, 256]


class ALTGVT(PCPVT):
    """
    alias Twins-SVT
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None,
                 num_heads=None, mlp_ratios=None, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=None, sr_ratios=None, block_cls=GroupBlock, wss=None):
        super(ALTGVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                     mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                     norm_layer, depths, sr_ratios, block_cls)
        if wss is None:
            wss = [7, 7, 7]
        if depths is None:
            depths = [4, 4, 4]
        if sr_ratios is None:
            sr_ratios = [4, 2, 1]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4]
        if num_heads is None:
            num_heads = [1, 2, 4]
        if embed_dims is None:
            embed_dims = [64, 128, 256]
        del self.blocks
        self.wss = wss
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks = nn.ModuleList()
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k], ws=1 if i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        self.apply(self._init_weights)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pcpvt_small_v0(pretrained=False, **kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pcpvt_base_v0(pretrained=False, **kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pcpvt_large_v0(pretrained=False, **kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def alt_gvt_small(pretrained=False, **kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def alt_gvt_base(pretrained=False, **kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)

    model.default_cfg = _cfg()
    return model


@register_model
def alt_gvt_large(pretrained=False, **kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)

    model.default_cfg = _cfg()
    return model
