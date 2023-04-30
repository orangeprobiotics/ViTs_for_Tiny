"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import torch.nn.functional as F

from gvt import PosCNN

from torch.nn import CrossEntropyLoss

from encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from attn import ProbAttention, AttentionLayer, FullAttention


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)  # （224，224）
        patch_size = (patch_size, patch_size)  # （16，16）
        self.img_size = img_size  # 224
        self.patch_size = patch_size  # 16
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # （14，14）
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, (H, W)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=12,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @[batch_size, num_heads, num_patches + 1, embed_dim_per_head]: multiply ->
        # [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):  # 对应的是Encoder Block 里面的MLP
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

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


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)  # MHA
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()  # Dropout，
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, factor=5, patch_size=16, in_c=3, num_classes=1000, e_layers=None,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, alpha=0.3, sparse=True, distil=True, stack=True, cls=False, PEG=False, sr_ratio=1):
        """
        Args:
            img_size (int, tuple): input image size 输入的图片尺寸
            patch_size (int, tuple): patch size patch尺寸
            in_c (int): number of input channels 输入的通道数
            num_classes (int): number of classes for classification head 分类的类别个数
            embed_dim (int): embedding dimension embedding维度
            depth (int): depth of transformer Transformer深度，就是Encoder的多少
            num_heads (int): number of attention heads 注意力头的个数
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim MLP Block 第一层全连接层的节点个数倍率
            qkv_bias (bool): enable bias for qkv if True 是否为qkv添加偏置
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set 假如设置了就在最后的MLP模块中有一个Pre-logits
            distilled (bool): model includes a distillation token and head as in DeiT models 是否像DeiT一样模型多一个蒸馏token和head
            drop_ratio (float): dropout rate dropout的概率
            attn_drop_ratio (float): attention dropout rate attention层里面的dropout率
            drop_path_ratio (float): stochastic depth rate MLP Block 中的dropout率
            embed_layer (nn.Module): patch embedding layer patch_embedding层
            norm_layer: (nn.Module): normalization layer 归一化层
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        if e_layers is None:
            # e_layers = [6, 4, 2]
            e_layers = [3, 2, 1]
        self.e_layers = e_layers
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if cls:
            self.num_tokens = 1
        else:
            self.num_tokens = 0
        if sparse:
            Attn = ProbAttention
        else:
            Attn = FullAttention
        inp_lens = list(range(len(e_layers)))
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        if cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        if PEG:
            self.EncoderBlock = EncoderLayer(AttentionLayer(Attn(factor, attention_dropout=attn_drop_ratio),
                                                            d_model=embed_dim, n_heads=num_heads, qkv_bias=qkv_bias,
                                                            sr_ratio=sr_ratio),
                                             d_model=embed_dim,
                                             d_ff=mlp_ratio * embed_dim,
                                             dropout=drop_ratio,
                                             activation=act_layer, )
            self.pos_block = PosCNN(embed_dim, embed_dim)
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_ratio, len(e_layers))]
        if stack:
            encoders = [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                Attn(factor, attention_dropout=attn_drop_ratio),
                                d_model=embed_dim, n_heads=num_heads, qkv_bias=qkv_bias),
                            d_model=embed_dim,
                            d_ff=mlp_ratio * embed_dim,
                            dropout=drop_ratio,
                            activation=act_layer,
                        ) for l in range(el)  # (0,1,2,0,1,0)
                    ],
                    [
                        ConvLayer(
                            embed_dim
                        ) for l in range(el - 1)  # (0,1,0)
                    ] if distil else None,
                    norm_layer=norm_layer(embed_dim)
                ) for el in e_layers]
            self.blocks = EncoderStack(encoders, inp_lens)
        else:
            self.blocks = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(factor, attention_dropout=attn_drop_ratio),
                            d_model=embed_dim, n_heads=num_heads, qkv_bias=qkv_bias),
                        d_model=embed_dim,
                        d_ff=mlp_ratio * embed_dim,
                        dropout=drop_ratio,
                        activation=act_layer
                    ) for l in range(depth)
                ],
                [
                    ConvLayer(
                        embed_dim
                    ) for l in range(depth - 1)
                ] if distil else None,
                norm_layer=norm_layer(embed_dim)
            )
        # if sparse:
        #     Attn = ProbAttention
        #     if stack:
        #         encoders = [
        #             Encoder(
        #                 [
        #                     EncoderLayer(
        #                         AttentionLayer(
        #                             Attn(factor, attention_dropout=attn_drop_ratio),
        #                             d_model=embed_dim, n_heads=num_heads, qkv_bias=qkv_bias),
        #                         d_model=embed_dim,
        #                         d_ff=mlp_ratio * embed_dim,
        #                         dropout=drop_ratio,
        #                         activation=act_layer,
        #                     ) for l in range(el)  # (0,1,2,0,1,0)
        #                 ],
        #                 [
        #                     ConvLayer(
        #                         embed_dim
        #                     ) for l in range(el - 1)
        #                 ] if distil else None,
        #                 norm_layer=torch.nn.LayerNorm(embed_dim)
        #             ) for el in e_layers]
        #         self.blocks = EncoderStack(encoders, inp_lens)
        #     else:
        #         self.blocks = Encoder(
        #             [
        #                 EncoderLayer(
        #                     AttentionLayer(
        #                         Attn(factor, attention_dropout=attn_drop_ratio),
        #                         d_model=embed_dim, n_heads=num_heads, qkv_bias=qkv_bias),
        #                     d_model=embed_dim,
        #                     d_ff=mlp_ratio * embed_dim,
        #                     dropout=drop_ratio,
        #                     activation=act_layer
        #                 ) for l in range(depth)
        #             ],
        #             [
        #                 ConvLayer(
        #                     embed_dim
        #                 ) for l in range(depth - 1)
        #             ] if distil else None,
        #             norm_layer=torch.nn.LayerNorm(embed_dim)
        #         )
        # else:
        #     inp_lens = list(range(len(e_layers)))
        #     if stack:
        #         # encoders = [Encoder(
        #         #     [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #         #            qk_scale=qk_scale,
        #         #            drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #         #            norm_layer=norm_layer, act_layer=act_layer)
        #         #      for i in range(el)],
        #         #     [
        #         #         ConvLayer(
        #         #             embed_dim
        #         #         ) for l in range(el - 1)
        #         #     ] if distil else None,
        #         #     norm_layer=torch.nn.LayerNorm(embed_dim)
        #         # )for el in e_layers]
        #         # self.blocks = EncoderStack(encoders, inp_lens)
        #         encoders = [
        #             Encoder(
        #                 [
        #                     EncoderLayer(
        #                         AttentionLayer(
        #                             Attn(factor, attention_dropout=attn_drop_ratio),
        #                             d_model=embed_dim, n_heads=num_heads, qkv_bias=qkv_bias),
        #                         d_model=embed_dim,
        #                         d_ff=mlp_ratio * embed_dim,
        #                         dropout=drop_ratio,
        #                         activation=act_layer
        #                     ) for l in range(el)  # (0,1,2,0,1,0)
        #                 ],
        #                 [
        #                     ConvLayer(
        #                         embed_dim
        #                     ) for l in range(el - 1)
        #                 ] if distil else None,
        #                 norm_layer=torch.nn.LayerNorm(embed_dim)
        #             ) for el in e_layers]
        #         self.blocks = EncoderStack(encoders, inp_lens)
        #     else:
        #         self.blocks = Encoder(
        #             [
        #                 EncoderLayer(
        #                     AttentionLayer(
        #                         Attn(factor, attention_dropout=attn_drop_ratio),
        #                         d_model=embed_dim, n_heads=num_heads, qkv_bias=qkv_bias),
        #                     d_model=embed_dim,
        #                     d_ff=mlp_ratio * embed_dim,
        #                     dropout=drop_ratio,
        #                     activation=act_layer
        #                 ) for l in range(depth)
        #             ],
        #             [
        #                 ConvLayer(
        #                     embed_dim
        #                 ) for l in range(depth - 1)
        #             ] if distil else None,
        #             norm_layer=torch.nn.LayerNorm(embed_dim)
        #         )
        #     # self.blocks = nn.Sequential(*[
        #     #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #     #           norm_layer=norm_layer, act_layer=act_layer)
        #     #     for i in range(depth)
        #     # ])
        self.norm = norm_layer(embed_dim)


        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x, (H, W) = self.patch_embed(x)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        if self.pos_embed is not None:
            x = self.pos_drop(x + self.pos_embed)
        else:
            x = self.pos_drop(x)
            x = self.EncoderBlock(x, H, W)
            x = self.pos_block(x, H, W)
        x = self.blocks(x, H, W)
        x = self.norm(x)
        if self.cls_token is not None:
            return self.pre_logits(x[:, 0])
        else:
            return self.pre_logits(x.mean(dim=1))

    def forward(self, x, labels=None):
        x1 = self.forward_features(x)
        logits = self.head(x1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            x2 = self.forward_features(x)
            newlogits = self.head(x2)
            loss1 = loss_fct(newlogits.view(-1, self.num_classes), labels.view(-1))
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
            return logits


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=6,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              drop_ratio=0.0,
                              attn_drop_ratio=0.0,
                              drop_path_ratio=0.1,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
