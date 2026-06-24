import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout(drop)

        # self.conv1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x

class Affine(nn.Module):
    def __init__(self, dim=256):
        super(Affine, self).__init__()

        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)

class ResMLP(nn.Module):
    def __init__(self, in_dim=256, mlp_ratio=2, hidden_dim=None, out_dim=None, seq_len=256, aff_first=True):
        super(ResMLP, self).__init__()

        if out_dim is None:
            out_dim = in_dim

        self.aff_first = aff_first

        self.norm1 = Affine(in_dim)
        self.linear = nn.Linear(seq_len, seq_len)
        self.norm2 = Affine(in_dim)
        if hidden_dim is None:
            self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=out_dim)
        else:
            self.mlp = Mlp(in_features=in_dim, hidden_features=hidden_dim, out_features=out_dim)

    def forward(self, x, residual_x=None):
        if residual_x is None:
            if self.aff_first:
                x = x + self.linear(self.norm1(x).transpose(1, 2)).transpose(1, 2)
            else:
                x = x + self.linear(x.transpose(1, 2)).transpose(1, 2)
        else:
            if self.aff_first:
                x = residual_x + self.linear(self.norm1(x).transpose(1, 2)).transpose(1, 2)
            else:
                x = residual_x + self.linear(x.transpose(1, 2)).transpose(1, 2)
        x = self.mlp(self.norm2(x))
        return x


class Prototype_Generation(nn.Module):
    def __init__(self, fea_channel=1024, num_tokens=256, seq_len=484):
        super(Prototype_Generation, self).__init__()

        self.num_tokens = num_tokens
        # self.norm1 = nn.LayerNorm(fea_channel + 1)
        self.norm1 = nn.LayerNorm(fea_channel)
        self.norm2 = nn.LayerNorm(fea_channel)
        # self.cluster_attn = nn.Sequential(
        #     nn.Conv2d(fea_channel+1, fea_channel // 4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
        #     nn.GELU()
        # )
        self.cluster_attn = nn.Sequential(
            nn.Conv2d(fea_channel, fea_channel // 4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU()
        )
        self.cluster_proj = nn.Linear(fea_channel // 4, num_tokens)

        self.affine1_first = Affine(fea_channel)
        self.affine2_first = Affine(fea_channel)
        self.resmlp_1 = ResMLP(in_dim=fea_channel, out_dim=fea_channel, seq_len=seq_len, aff_first=True)
        self.resmlp_2 = ResMLP(in_dim=fea_channel, out_dim=num_tokens, seq_len=seq_len, aff_first=True)

    def forward2(self, x, mask):
        weights = mask

        fg_cat = torch.cat((x, weights), dim=1)
        B, C, H, W = fg_cat.shape
        fg_cat = fg_cat.reshape(B, H*W, C)
        fg_cat = self.norm1(fg_cat)
        fg_cat = fg_cat.reshape(B, C, H, W)
        # map the high-dimensional input features to lower feature
        fg_cat = self.cluster_attn(fg_cat).reshape(B, H*W, -1)
        # fully-connected layer to map features to latent prototypes
        fg_cat = self.cluster_proj(fg_cat)
        fg_cat = fg_cat.permute(0, 2, 1)    # [B, num_tokens, H*W]
        

        r_mask = 1-mask
        x1 = x * mask
        x2 = x * r_mask
        x1 = x1.reshape(B, H*W, C-1)
        x2 = x2.reshape(B, H*W, C-1)
        x1 = self.affine1_first(x1)
        x2 = self.affine2_first(x2)
        x_sum = x1 + x2
        x_sum = self.resmlp_1(x_sum, x.reshape(B, H*W, C-1))
        x_sum = self.resmlp_2(x_sum).reshape(B, self.num_tokens, H*W)
        feat = x.reshape(B, H*W, C-1)

        fg_cat = fg_cat + x_sum
        # fg_cat = x_sum
        fg_cat = F.softmax(fg_cat, dim=-1)
        prototype = torch.einsum("...si,...id->...sd", fg_cat, feat)

        return prototype

    def forward(self, x, mask):
        weights = mask

        # fg_cat = torch.cat((x, weights), dim=1)
        fg_cat = x
        B, C, H, W = fg_cat.shape
        fg_cat = fg_cat.reshape(B, H*W, C)
        fg_cat = self.norm1(fg_cat)
        fg_cat = fg_cat.reshape(B, C, H, W)
        # map the high-dimensional input features to lower feature
        fg_cat = self.cluster_attn(fg_cat).reshape(B, H*W, -1)
        # fully-connected layer to map features to latent prototypes
        fg_cat = self.cluster_proj(fg_cat)
        fg_cat = fg_cat.permute(0, 2, 1)    # [B, num_tokens, H*W]
        

        r_mask = 1-mask
        x1 = x * mask
        x2 = x * r_mask
        x1 = x1.reshape(B, H*W, C)
        x2 = x2.reshape(B, H*W, C)
        x1 = self.affine1_first(x1)
        x2 = self.affine2_first(x2)
        x_sum = x1 + x2
        x_sum = self.resmlp_1(x_sum, x.reshape(B, H*W, C))
        x_sum = self.resmlp_2(x_sum).reshape(B, self.num_tokens, H*W)

        fg_cat = fg_cat + x_sum
        fg_cat = F.softmax(fg_cat, dim=-1)

        feat = x.reshape(B, H*W, C)
        prototype = torch.einsum("...si,...id->...sd", fg_cat, feat)

        return prototype