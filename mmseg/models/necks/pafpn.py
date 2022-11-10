# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .fpn import FPN
from torch.nn import init




class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=64, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual



class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:

            #B = x.data.size(0)
            #C = x.data.size(1)
            #H = x.data.size(2)
            #W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

@NECKS.register_module()
class PAN(nn.Module):
    # def __init__(self, inference=False):
    #     super().__init__()
    #     self.inference = inference
    #
    #     self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
    #     self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
    #     self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
    #     # SPP
    #     self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
    #     self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
    #     self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
    #
    #     # R -1 -3 -5 -6
    #     # SPP
    #     self.conv4 = Conv_Bn_Activation(256, 64, 1, 1, 'leaky')
    #     self.conv5 = Conv_Bn_Activation(64, 128, 3, 1, 'leaky')
    #     self.conv6 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
    #     # UP
    #     self.upsample1 = Upsample()
    #     # R 85
    #     self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
    #     # R -1 -3
    #     self.conv9 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
    #     self.conv10 = Conv_Bn_Activation(64, 128, 3, 1, 'leaky')
    #     self.conv11 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
    #     self.conv12 = Conv_Bn_Activation(64, 128, 3, 1, 'leaky')
    #     self.conv13 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
    #     self.conv14 = Conv_Bn_Activation(64, 32, 1, 1, 'leaky')
    #     # UP
    #     self.upsample2 = Upsample()
    #     # R 54
    #     self.conv15 = Conv_Bn_Activation(64, 32, 1, 1, 'leaky')
    #     # R -1 -3
    #     self.conv16 = Conv_Bn_Activation(64, 32, 1, 1, 'leaky')
    #     self.conv17 = Conv_Bn_Activation(32, 64, 3, 1, 'leaky')
    #     self.conv18 = Conv_Bn_Activation(64, 32, 1, 1, 'leaky')
    #     self.conv19 = Conv_Bn_Activation(32, 64, 3, 1, 'leaky')
    #     self.conv20 = Conv_Bn_Activation(64, 32, 1, 1, 'leaky')
    #
    #     self.conv21 = Conv_Bn_Activation(32, 64, 3, 2, 'leaky')
    #     self.conv22 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
    #     self.conv23 = Conv_Bn_Activation(64, 128, 3, 1, 'leaky')
    #     self.conv24 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
    #     self.conv25 = Conv_Bn_Activation(64, 128, 3, 1, 'leaky')
    #     self.conv26 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
    #     self.conv27 = Conv_Bn_Activation(64, 128, 3, 2, 'leaky')
    #     # *5 CBL
    #     self.conv28 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
    #     self.conv29 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
    #     self.conv30 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
    #     self.conv31 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
    #     self.conv32 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
    #
    #     self.conv33 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')

    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        # R -1 -3 -5 -6
        # SPP
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv4 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 256, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

        # UP

        self.convtrans1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))
        self.convtrans2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))
        self.convtrans3 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))
        self.convtrans4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))
        self.convtrans5 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True))
        self.convtrans6 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))
        self.convtrans7 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))
        self.convtrans8 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))

        # R 85
        self.conv8 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
        # R 54
        self.conv15 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(64, 128, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(64, 128, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')

        self.conv21 = Conv_Bn_Activation(64, 128, 3, 2, 'leaky')
        self.conv22 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv23 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv24 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv25 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv26 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv27 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')
        # *5 CBL
        self.conv28 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv29 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv30 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv31 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv32 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')

        self.conv33 = Conv_Bn_Activation(320, 256, 1, 1, 'leaky')
        self.conv34 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        #self.conv35 = Conv_Bn_Activation(128, 64, 1, 1, 'leaky')
        #att
        # self.conv35 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # self.cbam_256 = CBAMBlock(channel=256)

    @auto_fp16()
    def forward(self, inputs):
        dec_outs2 = inputs[1]
        # *3 conv
        dec4 = dec_outs2[0] # 512 32*32
        dec3 = dec_outs2[1] # 256 64*64
        dec2 = dec_outs2[2] # 128 128*128
        # *3 CBL
        x3 = self.conv3(dec4) #512-256
        x4 = self.conv4(x3) #256-512
        x5 = self.conv5(x4) #512-256 32*32 out
        # *1 CBL
        x6 = self.conv6(x5) #256-128 32*32

        # UP
        up = self.convtrans1(x6) # 128 64*64
        # *1 CBL
        x8 = self.conv8(dec3) # 256-128 64*64
        # enc_x5-enc_x4
        x8 = torch.cat([x8, up], dim=1) # 256 64*64
        # *5 CBL
        x9 = self.conv9(x8) # 256-128
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12) # 128 64*64 out
        # *1 CBL
        x14 = self.conv14(x13)# 64

        # UP
        up = self.convtrans2(x14) # 64 128*128
        # *1 CBL
        x15 = self.conv15(dec2) # 128-64 128*128
        # enc_x5-enc_x3
        x15 = torch.cat([x15, up], dim=1) # 128 128*128
        # *5 CBL
        x16 = self.conv16(x15) # 128-64
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19) # 64 128*128 out_1
        # *1 CBL
        x21 = self.conv21(x20) # 64-128 128*128->64*64
        # cat
        x21 = torch.cat([x21, x13], dim=1) #128+128 64*64
        # *5 CBL
        x22 = self.conv22(x21) # 256->128
        x23 = self.conv23(x22)
        x24 = self.conv24(x23)
        x25 = self.conv25(x24)
        x26 = self.conv26(x25) # 256-128 64*64 out_2
        # cat
        x27 = self.conv27(x26) # 128->256 64*64->32*32
        x27 = torch.cat([x27, x5], dim=1) #256+256 32*32
        # *5 CBL
        x28 = self.conv28(x27) #512->256
        x29 = self.conv29(x28)
        x30 = self.conv30(x29)
        x31 = self.conv31(x30)
        x32 = self.conv32(x31) #256 32*32 out_3

        dec_inputs = inputs[0]
        dec_x5 = dec_inputs[2] # 64 256*256

        #x20 = self.convtrans3(x20) #64-128 256*256

        x26 = self.convtrans4(x26) #128 128*128
        #x26 = self.convtrans4(x26) #128 256*256

        x32 = self.convtrans5(x32) #256 64*64
        x32 = self.convtrans6(x32) #256-128 128*128
        #x32 = self.convtrans7(x32) #128 256*256

        cat_out = torch.cat([x20, x26, x32], dim=1) #320 128*128
        cat_out = self.conv33(cat_out)
        cat_out = self.conv34(cat_out) # 256-128
        cat_out = self.convtrans8(cat_out) #64 256*256
        #cat_out = self.conv35(cat_out) # 64
        #dec_x5_128 = self.conv21(dec_x5) #128 128*128
        cat_out = torch.cat([dec_x5, cat_out], dim=1) #128

        outs = [cat_out, dec_x5]

        #return x20 x13 x5
        return tuple(outs)



        # # enc_inputs = inputs[1]
        # # # *3 conv
        # # enc_x5 = enc_inputs[2] # 64 256*256
        # # enc_x4 = enc_inputs[1] # 128 128*128
        # # enc_x3 = enc_inputs[0] # 256 64*64
        # dec_inputs = inputs[0]
        # dec_x5 = dec_inputs[2]  # 64 256*256
        # dec_x4 = dec_inputs[1] # 128 128*128
        # dec_x3 = dec_inputs[0] # 256 64*64
        # # *2 CBL
        # x4 = self.conv4(dec_x3) #256-64
        # x5 = self.conv5(x4) #64-128 64*64 out
        # # *1 CBL
        # x6 = self.conv6(x5) #128-64 64*64
        #
        # # UP
        # up = self.upsample1(x6, dec_x4.size(), self.inference) # 64 128*128
        # # *1 CBL
        # x8 = self.conv8(dec_x4) # 128-64 128*128
        # # enc_x5-enc_x4
        # x8 = torch.cat([x8, up], dim=1) # 128 128*128
        # # *5 CBL
        # x9 = self.conv9(x8) # 128-64
        # x10 = self.conv10(x9)
        # x11 = self.conv11(x10)
        # x12 = self.conv12(x11)
        # x13 = self.conv13(x12) # 64 128*128 out
        # # *1 CBL
        # x14 = self.conv14(x13)# 32
        #
        # # UP
        # up = self.upsample2(x14, dec_x5.size(), self.inference) # 32 256*256
        # # *1 CBL
        # x15 = self.conv15(dec_x5) # 64-32 256*256
        # # enc_x5-enc_x3
        # x15 = torch.cat([x15, up], dim=1) # 64 256*256
        # # *5 CBL
        # x16 = self.conv16(x15) # 64-32
        # x17 = self.conv17(x16)
        # x18 = self.conv18(x17)
        # x19 = self.conv19(x18)
        # x20 = self.conv20(x19) # 32 256*256 out_1
        # # *1 CBL
        # x21 = self.conv21(x20) # 64 256*256->128*128
        # # cat
        # x21 = torch.cat([x21, x13], dim=1) #64+64 128*128
        # # *5 CBL
        # x22 = self.conv22(x21) # 128->64
        # x23 = self.conv23(x22)
        # x24 = self.conv24(x23)
        # x25 = self.conv25(x24)
        # x26 = self.conv26(x25) # 64 128*128 out_2
        # # cat
        # x27 = self.conv27(x26) # 64->128 128*128->64*64
        # x27 = torch.cat([x27, x5], dim=1) #128+128 64*64
        # # *5 CBL
        # x28 = self.conv28(x27)
        # x29 = self.conv29(x28)
        # x30 = self.conv30(x29)
        # x31 = self.conv31(x30)
        # x32 = self.conv32(x31) #128 64*64 out_3
        #
        # x26 = self.upsample1(x26, dec_x5.size(), self.inference) # 64 128*128->256*256
        # x32 = self.upsample1(x32, dec_x5.size(), self.inference) # 128 64*64->256*256
        # cat_out = torch.cat([x20, x26, x32],dim=1) # 32+64+128=224 256*256
        # # dec_inputs = inputs[0]
        # # dec_x5 = dec_inputs[2] # 64 256*256
        # # dec_x4 = dec_inputs[1] # 128 128*128
        # # dec_x3 = dec_inputs[0] # 256 64*64
        # # x20 = self.conv19(x20) # 32->64
        # # x26 = self.conv25(x26) # 64->128
        # # x32 = self.conv31(x32) # 128->256
        #
        # # cat_out1 = torch.cat([dec_x5, x20], dim = 1) #128 256*256
        # # cat_out1 = self.conv26(cat_out1) # 64
        # # cat_out2 = torch.cat([dec_x4, x26], dim = 1) #256 128*128
        # # cat_out2 = self.conv32(cat_out2) # 128
        # # cat_out3 = torch.cat([dec_x3, x32], dim = 1) #512 64*64
        # # cat_out3 = self.conv33(cat_out3) #256
        #
        # outs = [cat_out, dec_x5]
        #
        # #return x20 x13 x5
        # return tuple(outs)







# class PAFPN(FPN):
#     """Path Aggregation Network for Instance Segmentation.
#
#     This is an implementation of the `PAFPN in Path Aggregation Network
#     <https://arxiv.org/abs/1803.01534>`_.
#
#     Args:
#         in_channels (List[int]): Number of input channels per scale.
#         out_channels (int): Number of output channels (used at each scale)
#         num_outs (int): Number of output scales.
#         start_level (int): Index of the start input backbone level used to
#             build the feature pyramid. Default: 0.
#         end_level (int): Index of the end input backbone level (exclusive) to
#             build the feature pyramid. Default: -1, which means the last level.
#         add_extra_convs (bool | str): If bool, it decides whether to add conv
#             layers on top of the original feature maps. Default to False.
#             If True, it is equivalent to `add_extra_convs='on_input'`.
#             If str, it specifies the source feature map of the extra convs.
#             Only the following options are allowed
#
#             - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
#             - 'on_lateral':  Last feature map after lateral convs.
#             - 'on_output': The last output feature map after fpn convs.
#         relu_before_extra_convs (bool): Whether to apply relu before the extra
#             conv. Default: False.
#         no_norm_on_lateral (bool): Whether to apply norm on lateral.
#             Default: False.
#         conv_cfg (dict): Config dict for convolution layer. Default: None.
#         norm_cfg (dict): Config dict for normalization layer. Default: None.
#         act_cfg (str): Config dict for activation layer in ConvModule.
#             Default: None.
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#     """
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_outs,
#                  start_level=0,
#                  end_level=-1,
#                  add_extra_convs=False,
#                  relu_before_extra_convs=False,
#                  no_norm_on_lateral=False,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  act_cfg=None,
#                  init_cfg=dict(
#                      type='Xavier', layer='Conv2d', distribution='uniform')):
#         super(PAFPN, self).__init__(
#             in_channels,
#             out_channels,
#             num_outs,
#             start_level,
#             end_level,
#             add_extra_convs,
#             relu_before_extra_convs,
#             no_norm_on_lateral,
#             conv_cfg,
#             norm_cfg,
#             act_cfg,
#             init_cfg=init_cfg)
#         # add extra bottom up pathway
#         self.downsample_convs = nn.ModuleList()
#         self.pafpn_convs = nn.ModuleList()
#         for i in range(self.start_level + 1, self.backbone_end_level):
#             d_conv = ConvModule(
#                 out_channels,
#                 out_channels,
#                 3,
#                 stride=2,
#                 padding=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#                 inplace=False)
#             pafpn_conv = ConvModule(
#                 out_channels,
#                 out_channels,
#                 3,
#                 padding=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#                 inplace=False)
#             self.downsample_convs.append(d_conv)
#             self.pafpn_convs.append(pafpn_conv)
#
#     @auto_fp16()
#     def forward(self, inputs):
#         """Forward function."""
#         #assert len(inputs) == len(self.in_channels)
#         dec_inputs = inputs[0]
#         enc_inputs = inputs[1]
#
#         # build lateralspafpn结构图
#         laterals = [
#             lateral_conv(dec_inputs[i + self.start_level])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]
#         #origin_laterals = laterals
#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             prev_shape = laterals[i - 1].shape[2:]
#             laterals[i - 1] += F.interpolate(
#                 laterals[i], size=prev_shape, mode='nearest')
#
#         # build outputs
#         # part 1: from original levels
#         inter_outs = [
#             self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
#         ]
#
#         #逆序
#         inter_outs.reverse()
#
#         # part 2: add bottom-up path
#         for i in range(0, used_backbone_levels - 1):
#             inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])
#
#         #逆序
#         inter_outs.reverse()
#
#         outs = []
#         outs.append(inter_outs[0])
#         outs.extend([
#             self.pafpn_convs[i - 1](inter_outs[i])
#             for i in range(1, used_backbone_levels)
#         ])
#         for i in range(len(outs)-1):
#             outs[i] = torch.cat([enc_inputs[i],outs[i]], 1)
#
#         # part 3: add extra levels
#         if self.num_outs > len(outs):
#             # use max pool to get more levels on top of outputs
#             # (e.g., Faster R-CNN, Mask R-CNN)
#             if not self.add_extra_convs:
#                 for i in range(self.num_outs - used_backbone_levels):
#                     outs.append(F.max_pool2d(outs[-1], 1, stride=2))
#             # add conv layers on top of original feature maps (RetinaNet)
#             else:
#                 if self.add_extra_convs == 'on_input':
#                     orig = dec_inputs[self.backbone_end_level - 1]
#                     outs.append(self.fpn_convs[used_backbone_levels](orig))
#                 elif self.add_extra_convs == 'on_lateral':
#                     outs.append(self.fpn_convs[used_backbone_levels](
#                         laterals[-1]))
#                 elif self.add_extra_convs == 'on_output':
#                     outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
#                 else:
#                     raise NotImplementedError
#                 for i in range(used_backbone_levels + 1, self.num_outs):
#                     if self.relu_before_extra_convs:
#                         outs.append(self.fpn_convs[i](F.relu(outs[-1])))
#                     else:
#                         outs.append(self.fpn_convs[i](outs[-1]))
#         return tuple(outs)
