import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from timm.models.layers import trunc_normal_
from lib.backbone.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.backbone.pvt_v2 import pvt_v2_b2, pvt_v2_b5

from lib.sentry.prototype_gen import Prototype_Generation
from lib.sentry.sentry_discriminator import Discriminator

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout(drop)

        self.conv1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)

        self.apply(self._init_weights)
    
    def dwconv(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)
        return x

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

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        if H is not None and W is not None:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x

class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)

        return output

"""
Source Code Reference: PNS+
Paper & Project: https://github.com/GewelsJI/VPS
"""
class combine_feature(nn.Module):
    def __init__(self, high_channel=128, low_channel=128, middle_channel=32):
        super(combine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(high_channel, middle_channel)
        self.up2_low = nn.Conv2d(low_channel, middle_channel, 1, stride=1, padding=0, bias=False)
        self.up2_bn2 = nn.BatchNorm2d(middle_channel)
        self.up2_act = nn.PReLU(middle_channel)
        self.refine = nn.Sequential(
            nn.Conv2d(middle_channel, middle_channel, 3, padding=1, bias=False), 
            nn.BatchNorm2d(middle_channel), nn.PReLU())

    def forward(self, high_fea, low_fea=None):
        high_fea = self.up2_high(high_fea)
        if low_fea is not None:
            low_fea = self.up2_bn2(self.up2_low(low_fea))
            refine_feature = self.refine(self.up2_act(high_fea + low_fea))
        else:
            refine_feature = self.refine(self.up2_act(high_fea))
        return refine_feature

class SentryNet(nn.Module):
    def __init__(self, f_num=5, img_size=(224, 224), mlp_ratio=2.0, **kwargs):
        super(SentryNet, self).__init__()

        self.f_num = f_num

        #### Res2Net-50 ####
        # self.feature_extractor = res2net50_v1b_26w_4s(pretrained=True)
        # self.ext_type = "res2net"
        # self.fea_channels = [256, 512, 1024, 2048]
        #### PVT-V2-B2 ####
        self.feature_extractor = pvt_v2_b2()
        self.load_model("/media/cgl/pretrained/pvt_v2_b2.pth")
        self.ext_type = "pvtv2"
        self.fea_channels = [64, 128, 320, 512]

        middle_channel = 32
        self.decoder = combine_feature(self.fea_channels[1], self.fea_channels[0], middle_channel)
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(middle_channel, 1, kernel_size=1, bias=False))

        ### for prototypes ###
        self.ds_H = img_size[0] // 16
        self.ds_W = img_size[1] // 16
        self.tokens_H = self.ds_H // 2
        self.tokens_W = self.ds_W // 2
        self.num_tokens = (self.tokens_H) * (self.tokens_W)
        p_in_channels = self.fea_channels[2]
        p_out_channels = self.fea_channels[3]
        self.prototype_gen = Prototype_Generation(fea_channel=p_in_channels, num_tokens=self.num_tokens, seq_len=self.ds_H*self.ds_W)
        
        ### sentry training ###
        self.temp_rate = (0.9, 1.1)
        self.linear_f2p = nn.Linear(self.ds_H * self.ds_W, self.num_tokens)

        ### memory readout ###
        self.prototype_linear = nn.Linear(p_in_channels, p_in_channels)
        self.p_linear_align = nn.Linear(p_in_channels, p_out_channels)

        self.final_decoder = combine_feature(self.fea_channels[3], self.fea_channels[2], middle_channel)
        self.final_SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(middle_channel, 1, kernel_size=1, bias=False))

        self.memory_p = None
        self.memory_neg_p = None

        # self.sentry = Discriminator(in_channels=1, pool_size=11).cuda()

    def load_model(self, ckpt):
        pretrained_dict = torch.load(ckpt)
        model_dict = self.feature_extractor.state_dict()
        print("Load pretrained parameters from {}".format(ckpt))
        # for k, v in pretrained_dict.items():
        #     if (k in model_dict):
        #         print("load:%s"%k)
        #     else:
        #         print("jump over:%s"%k)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.feature_extractor.load_state_dict(model_dict)
        print("PVTv2 Loaded!")

    def feature_extract(self, x):
        if self.ext_type == "res2net":
            x = self.feature_extractor.conv1(x)
            x = self.feature_extractor.bn1(x)
            x = self.feature_extractor.relu(x)
            x = self.feature_extractor.maxpool(x)
            x1 = self.feature_extractor.layer1(x)
            x2 = self.feature_extractor.layer2(x1)
            x3 = self.feature_extractor.layer3(x2)
            x4 = self.feature_extractor.layer4(x3)
        elif self.ext_type == "pvtv2":
            x1, x2, x3, x4 = self.feature_extractor(x)
        return x1, x2, x3, x4

    def similarity_encode(self, p1=None, p2=None, f1=None, f2=None):
        assert (p1 is not None and p2 is not None) or (f1 is not None and f2 is not None), "Similarity Compution needs prototype pairs or focused maps pairs!"
        if p1 is not None and p2 is not None:
            sim_pro = F.cosine_similarity(p1, p2, dim=-1).unsqueeze(1)
        if f1 is not None and f2 is not None:
            sim_focus = F.cosine_similarity(f1, f2, dim=-1).unsqueeze(1)

        if p1 is not None and f1 is not None:
            sim_focus2pro = self.linear_f2p(sim_focus)
            # cat: [B, 2 self.num_tokens] -> [B*(T-1), 2, H//16//2, W//16//2]
            sim_cat = torch.cat((sim_pro, sim_focus2pro), dim=1)
            sim_cat = sim_cat.reshape(sim_cat.shape[0], 2, self.tokens_H, self.tokens_W)
        elif p1 is not None:
            sim_cat = sim_pro.reshape(sim_pro.shape[0], 1, self.tokens_H, self.tokens_W)
        elif f1 is not None:
            sim_cat = sim_focus.reshape(sim_focus.shape[0], 1, self.tokens_H, self.tokens_W)

        return sim_cat

    def corr_calculate(self, x, prototype=None):
        assert prototype is not None, "Correlationship Calculation needs prototype!"
        if prototype is not None:
            x_B, x_C, x_H, x_W = x.shape
            p_B, num_tokens, p_C = prototype.shape
            if x_C != p_C:
                prototype = self.p_linear_align(prototype)
            n_p = prototype / prototype.norm(dim=2, keepdim=True)
            n_x = x.view(x_B, x_C, x_H*x_W) / x.view(x_B, x_C, x_H*x_W).norm(dim=1, keepdim=True)
            corr = torch.bmm(n_p, n_x)      # [B, N, x_HxW]
            return corr.permute(0, 2, 1), prototype
            # return corr, prototype
    
    def forward_generate(self, x):
        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])
        x1, x2, x3, x4 = self.feature_extract(x)

        ### initial coarse mask ###
        # x4_up = F.interpolate(x4, size=(x3.shape[-2], x3.shape[-1]), mode="bilinear", align_corners=False)
        # decoder_out = self.decoder(x4_up, x3)
        x2_up = F.interpolate(x2, size=(x1.shape[-2], x1.shape[-1]), mode="bilinear", align_corners=False)
        decoder_out = self.decoder(x2_up, x1)
        # x3_up = F.interpolate(x3, size=(x2.shape[-2], x2.shape[-1]), mode="bilinear", align_corners=False)
        # decoder_out = self.decoder(x3_up, x2)
        decoder_out = self.SegNIN(decoder_out)
        # self.initial_mask = decoder_out
        initial_out = F.interpolate(decoder_out, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
        
        ##### Prototype Generation #####
        mask_for_pro = F.interpolate(decoder_out, size=(x3.shape[-2], x3.shape[-1]), mode="bilinear", align_corners=False)
        prototypes = self.prototype_gen(x3, torch.sigmoid(mask_for_pro))      # [B, num_tokens, x3.C]

        prototypes = prototypes.reshape(origin_shape[0], origin_shape[1], self.num_tokens, x3.shape[1])

        features = [x1, x2, x3, x4]
        return prototypes, features, initial_out

    def forward_train_sentry(self, prototypes):
        reset_mem = False
        ## reset
        if self.memory_neg_p is None or self.memory_p.shape[0] != prototypes.shape[0]:
            # self.memory_neg_p = copy.deepcopy(prototypes[:, 0, :, :].detach())
            self.memory_neg_p = copy.deepcopy(prototypes.detach())
            reset_mem = True

        positive_pairs = None
        negative_pairs = None
        for i in range(self.f_num-1):
            # negative pairs
            if reset_mem:
                if i == 0:
                    sim_cat_n = self.similarity_encode(
                        self.memory_neg_p[:, 0, :, :].detach(), prototypes[:, self.f_num-1, :, :].detach()
                    )
                    negative_pairs = sim_cat_n
                else:
                    temp = random.uniform(self.temp_rate[0], self.temp_rate[1])
                    new_neg_pairs = sim_cat_n.clone() * temp
                    new_neg_pairs = torch.clamp(new_neg_pairs, min=-1.0, max=1.0)
                    negative_pairs = torch.cat((negative_pairs, new_neg_pairs), dim=0)
            else:
                sim_cat_n = self.similarity_encode(
                    self.memory_neg_p[:, i, :, :].detach(), prototypes[:, self.f_num-1, :, :].detach()
                )
                if negative_pairs is None:
                    negative_pairs = sim_cat_n
                else:
                    negative_pairs = torch.cat((negative_pairs, sim_cat_n), dim=0)

            # positive pairs
            sim_cat = self.similarity_encode(
                prototypes[:, i, :, :].detach(), prototypes[:, i+1, :, :].detach()
            )
            if positive_pairs is None:
                positive_pairs = sim_cat
            else:
                positive_pairs = torch.cat((positive_pairs, sim_cat), dim=0)

        ## update
        self.memory_neg_p = copy.deepcopy(prototypes.detach())

        # neg_out = self.sentry(negative_pairs.detach())
        # pos_out = self.sentry(positive_pairs.detach())
        # return neg_out, pos_out
        return negative_pairs, positive_pairs
    
    def forward(self, prototypes, features, gt_sizes=(352, 352), mode="train", sentry=None):
        if mode == "train":
            self.memory_p = None

        x1, x2, x3, x4 = features
        bs = x1.shape[0] // self.f_num
        x_f = x4.reshape(bs, self.f_num, x4.shape[1], x4.shape[2], x4.shape[3])
        assert sentry is not None, "Model Inference Needs Sentry!"
        new_x_f = torch.zeros(x_f.shape, requires_grad=True).to(x_f.device)

        ### Frame-wise ###
        related_threshold = 0.5
        for i in range(0, self.f_num):
            if self.memory_p is None:
                self.memory_p = prototypes[:, 0].detach()
            else:
                current_prototype = prototypes[:, i]
                sim_mem = self.similarity_encode(
                   current_prototype , self.memory_p
                )
                label_mem = sentry(sim_mem).detach()

                # long-term information processing #
                m_B, m_N, m_C = self.memory_p.shape

                sim_mask_p = sim_mem.flatten(2, 3).permute(0, 2, 1)
                fused_prototype = self.memory_p + sim_mask_p * self.prototype_linear(current_prototype)

                self.memory_p[label_mem > related_threshold] = fused_prototype[label_mem > related_threshold].clone().detach()
                # replace: those unrelated prototype and focused map are replaced together
                self.memory_p[label_mem <= related_threshold] = current_prototype[label_mem <= related_threshold].clone().detach()

                ## long-term temporal propagation ##
                corr_p, new_p = self.corr_calculate(x_f[:, i], prototype=fused_prototype.detach())    # corr: [B, HxW, N]  new_p: [B, N, x_f_C]
                corr_p_s = F.softmax(corr_p, dim=-1)
                
                new_p = torch.bmm(corr_p_s, new_p).reshape(bs, new_x_f.shape[2], new_x_f.shape[3], new_x_f.shape[4])
                new_x_f[:, i][label_mem > related_threshold] += new_p[label_mem > related_threshold]

                ## dynamic divergence propagation ##
                corr_p_1, _ = self.corr_calculate(x_f[:, i], prototype=prototypes[:, i].detach())
                if i == 0:
                    corr_p_2, _ = self.corr_calculate(x_f[:, i], prototype=prototypes[:, i+1].detach())
                else:
                    corr_p_2, _ = self.corr_calculate(x_f[:, i], prototype=prototypes[:, i-1].detach())
                corr_p_s1 = F.softmax(corr_p_1, dim=-1)
                corr_p_s2 = F.softmax(corr_p_2, dim=-1)
                dynamic_map = torch.abs(corr_p_s1 - corr_p_s2)
                
                dynamic_f = torch.bmm(dynamic_map, x_f[:, i].flatten(-2).permute(0, 2, 1)).reshape(bs, new_x_f.shape[2], new_x_f.shape[3], new_x_f.shape[4])
                new_x_f[:, i] += dynamic_f

        x_f = x_f + new_x_f
        x_f = x_f.view(-1, x_f.shape[-3], x_f.shape[-2], x_f.shape[-1])
        x_f_up = F.interpolate(x_f, size=(x3.shape[-2], x3.shape[-1]), mode="bilinear", align_corners=False)
        decoder_out = self.final_decoder(x_f_up, x3)
        decoder_out = self.final_SegNIN(decoder_out)
        final_out = F.interpolate(decoder_out, size=(gt_sizes[0], gt_sizes[1]), mode="bilinear", align_corners=False)

        assert mode in ['train', 'eval'], "mode should be train or eval."
        if mode == "train":
            return final_out
        else:
            return torch.sigmoid(final_out)