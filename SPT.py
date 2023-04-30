import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import math
import torch.nn.functional
from timm.models.layers import to_2tuple


class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_dim, dim, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.patch_shifting = PatchShifting(patch_size)

        # patch_dim = (in_dim * 5) * (patch_size ** 2)
        patch_dim = in_dim * (patch_size ** 2)

        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        H, W = H // self.patch_size, W // self.patch_size
        out = x
        out = self.patch_shifting(out)
        out = self.merging(out)

        return out, (H, W)


class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * (1 / 2))

    def forward(self, x):
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        # if self.is_mean:
        #     x_pad = x_pad.mean(dim=1, keepdim = True)

        """ 4 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1) 
        #############################

        """ 4 diagonal directions """
        # #############################
        x_lu = x_pad[:, :, :-self.shift * 2, :-self.shift * 2]
        x_ru = x_pad[:, :, :-self.shift * 2, self.shift * 2:]
        x_lb = x_pad[:, :, self.shift * 2:, :-self.shift * 2]
        x_rb = x_pad[:, :, self.shift * 2:, self.shift * 2:]
        # x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)
        x_cat = x + x_lu + x_ru + x_rb + x_lb
        # #############################

        """ 8 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        # x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        # x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        # x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2, x_lu, x_ru, x_lb, x_rb], dim=1) 
        #############################

        # out = self.out(x_cat)
        out = x_cat
        # out = x_cat/4

        return out
