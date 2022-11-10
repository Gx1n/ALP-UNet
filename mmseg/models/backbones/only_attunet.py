# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils.parrots_wrapper import _BatchNorm
from ..builder import BACKBONES

#新增注意力机制
import torch
from torch.nn import init


# class Mish(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         x = x * (torch.tanh(torch.nn.functional.softplus(x)))
#         return x
#
#
# class Upsample(nn.Module):
#     def __init__(self):
#         super(Upsample, self).__init__()
#
#     def forward(self, x, target_size, inference=False):
#         assert (x.data.dim() == 4)
#         # _, _, tH, tW = target_size
#
#         if inference:
#
#             #B = x.data.size(0)
#             #C = x.data.size(1)
#             #H = x.data.size(2)
#             #W = x.data.size(3)
#
#             return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
#                     expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
#                     contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
#         else:
#             return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')
#
#
# class Conv_Bn_Activation(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
#         super().__init__()
#         pad = (kernel_size - 1) // 2
#
#         self.conv = nn.ModuleList()
#         if bias:
#             self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
#         else:
#             self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
#         if bn:
#             self.conv.append(nn.BatchNorm2d(out_channels))
#         if activation == "mish":
#             self.conv.append(Mish())
#         elif activation == "relu":
#             self.conv.append(nn.ReLU(inplace=True))
#         elif activation == "leaky":
#             self.conv.append(nn.LeakyReLU(0.1, inplace=True))
#         elif activation == "linear":
#             pass
#         # else:
#         #     print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
#         #                                                sys._getframe().f_code.co_name, sys._getframe().f_lineno))
#
#     def forward(self, x):
#         for l in self.conv:
#             x = l(x)
#         return x
#
#
# class ResBlock(nn.Module):
#     """
#     Sequential residual blocks each of which consists of \
#     two convolution layers.
#     Args:
#         ch (int): number of input and output channels.
#         nblocks (int): number of residual blocks.
#         shortcut (bool): if True, residual tensor addition is enabled.
#     """
#
#     def __init__(self, ch, nblocks=1, shortcut=True):
#         super().__init__()
#         self.shortcut = shortcut
#         self.module_list = nn.ModuleList()
#         for i in range(nblocks):
#             resblock_one = nn.ModuleList()
#             resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
#             resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
#             self.module_list.append(resblock_one)
#
#     def forward(self, x):
#         for module in self.module_list:
#             h = x
#             for res in module:
#                 h = res(h)
#             x = x + h if self.shortcut else h
#         return x

# class DownSample1(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
#
#         self.conv2 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
#         self.conv3 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
#         # [route]
#         # layers = -2
#         self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
#
#         self.conv5 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
#         self.conv6 = Conv_Bn_Activation(64, 128, 3, 1, 'mish')
#         # [shortcut]
#         # from=-3
#         # activation = linear
#
#         self.conv7 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
#         # [route]
#         # layers = -1, -7
#         self.conv8 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
#
#     def forward(self, input):
#         x1 = self.conv1(input)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         # route -2
#         x4 = self.conv4(x2)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         # shortcut -3
#         x6 = x6 + x4
#
#         x7 = self.conv7(x6)
#         # [route]
#         # layers = -1, -7
#         x7 = torch.cat([x7, x3], dim=1)
#         x8 = self.conv8(x7) # 128 128*128
#         return x8
#
#
# class DownSample2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
#         self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
#         # r -2
#         self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
#
#         self.resblock = ResBlock(ch=128, nblocks=2)
#
#         # s -3
#         self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
#         # r -1 -10
#         self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
#
#     def forward(self, input):
#         x1 = self.conv1(input)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x1)
#
#         r = self.resblock(x3)
#         x4 = self.conv4(r)
#
#         x4 = torch.cat([x4, x2], dim=1)
#         x5 = self.conv5(x4) # 256 64*64
#         return x5
#
#
# class DownSample3(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
#         self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
#         self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
#
#         self.resblock = ResBlock(ch=256, nblocks=8)
#         self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
#         self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
#
#     def forward(self, input):
#         x1 = self.conv1(input)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x1)
#
#         r = self.resblock(x3)
#         x4 = self.conv4(r)
#
#         x4 = torch.cat([x4, x2], dim=1)
#         x5 = self.conv5(x4) #512 32*32
#         return x5
#
#
# class DownSample4(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
#         self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
#         self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
#
#         self.resblock = ResBlock(ch=512, nblocks=8)
#         self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
#         self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')
#
#     def forward(self, input):
#         x1 = self.conv1(input)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x1)
#
#         r = self.resblock(x3)
#         x4 = self.conv4(r)
#
#         x4 = torch.cat([x4, x2], dim=1)
#         x5 = self.conv5(x4) # 1024 16*16
#         return x5


class WSConv2d(nn.Conv2d):
    def __init___(self, in_channels, out_channels, kernel_size, stride=1,
        padding=1, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1,1,1,1) + 1e-5
        #std = torch.sqrt(torch.var(weight.view(weight.size(0),-1),dim=1)+1e-12).view(-1,1,1,1)+1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, 1, self.dilation, self.groups)


def conv_ws(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return WSConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


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

# ASPP Module
class myConv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1,
                    padding=0, dilation=1, bias=True, norm='GN', act='ELU', num_groups=32):
        super(myConv, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        else:
            act = nn.ReLU(True)
            #act = nn.GELU()
        module = []
        if norm == 'GN':
            module.append(nn.GroupNorm(num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(in_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
        module.append(act)
        module.append(conv(in_ch, out_ch, kernel_size=kSize, stride=stride,
                            padding=padding, dilation=dilation, groups=1, bias=bias))
        self.module = nn.Sequential(*module)
    def forward(self, x):
        out = self.module(x)
        return out

class Dilated_bottleNeck(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck, self).__init__()
        conv = conv_ws
        # in feat = 1024 in ResNext101 and ResNet101
        self.reduction1 = conv(in_feat, in_feat//2, kernel_size=1, stride = 1, bias=False, padding=0)
        self.aspp_d3 = nn.Sequential(myConv(in_feat//2, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=3, dilation=3,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d6 = nn.Sequential(myConv(in_feat//2 + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2 + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=6, dilation=6,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d12 = nn.Sequential(myConv(in_feat, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=12, dilation=12,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d18 = nn.Sequential(myConv(in_feat + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=18, dilation=18,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.reduction2 = myConv(((in_feat//4)*4) + (in_feat//2), in_feat, kSize=3, stride=1, padding=1,bias=False, norm=norm, act=act, num_groups = ((in_feat//4)*4 + (in_feat//2))//16)
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3],dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6],dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12],dim=1)
        d18 = self.aspp_d18(cat3)
        out = self.reduction2(torch.cat([x,d3,d6,d12,d18], dim=1))
        return out      # 512 x H/16 x W/16

class Dilated_bottleNeck2(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck2, self).__init__()
        conv = conv_ws
        # in feat = 1024 in ResNext101 and ResNet101
        #self.reduction1 = conv(in_feat, in_feat//2, kernel_size=1, stride = 1, bias=False, padding=0)
        self.reduction1 = conv(in_feat, in_feat//2, kernel_size=3, stride = 1, padding=1, bias=False)
        self.aspp_d3 = nn.Sequential(myConv(in_feat//2, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=3, dilation=3,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d6 = nn.Sequential(myConv(in_feat//2 + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2 + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=6, dilation=6,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d12 = nn.Sequential(myConv(in_feat, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=12, dilation=12,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d18 = nn.Sequential(myConv(in_feat + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=18, dilation=18,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d24 = nn.Sequential(myConv(in_feat + in_feat//2, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat + in_feat//2)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=24, dilation=24,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.reduction2 = myConv(((in_feat//4)*5) + (in_feat//2), in_feat//2, kSize=3, stride=1, padding=1,bias=False, norm=norm, act=act, num_groups = ((in_feat//4)*5 + (in_feat//2))//16)
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3],dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6],dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12],dim=1)
        d18 = self.aspp_d18(cat3)
        cat4 = torch.cat([cat3, d18],dim=1)
        d24 = self.aspp_d24(cat4)
        out = self.reduction2(torch.cat([x,d3,d6,d12,d18,d24], dim=1))
        return out      # 512 x H/16 x W/16

#Unet==========================================================================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            #nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)

class wsDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.wsdouble_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            #nn.GELU(),
            WSConv2d(in_channels=in_channels,out_channels=mid_channels, kernel_size=3),
            #nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            #nn.GELU(),
            WSConv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3)
            #nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.wsdouble_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = wsDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, (in_channels // 2), kernel_size=2, stride=2)
            self.conv = wsDoubleConv(in_channels, out_channels)

    #新增差异
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@BACKBONES.register_module()
class AttUNet1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, norm_eval=False):
        super(AttUNet1, self).__init__()
        self.norm_eval = norm_eval
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # 注意力
        self.cbam_64 = CBAMBlock(channel=64)
        self.cbam_128 = CBAMBlock(channel=128)
        self.cbam_256 = CBAMBlock(channel=256)
        self.cbam_512 = CBAMBlock(channel=512)
        self.cbam_1024 = CBAMBlock(channel=1024)
        #ASPP
        # self.ASPP_64 = Dilated_bottleNeck(norm=None, act=None, in_feat=64)
        # self.ASPP_128 = Dilated_bottleNeck(norm=None, act=None, in_feat=128)
        # self.ASPP_256 = Dilated_bottleNeck(norm=None, act=None, in_feat=256)
        # self.ASPP_512 = Dilated_bottleNeck(norm=None, act=None, in_feat=512)

        #CSP模块
        # self.down1 = DownSample1()
        # self.down2 = DownSample2()
        # self.down3 = DownSample3()
        # self.down4 = DownSample4()
        # self.conv1 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        # self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        # self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        # self.conv4 = Conv_Bn_Activation(2048, 1024, 1, 1, 'mish')

    def forward(self, x):
        #计算差异
        rgb_down2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        rgb_down4 = F.interpolate(rgb_down2, scale_factor=0.5, mode='bilinear')
        rgb_down8 = F.interpolate(rgb_down4, scale_factor=0.5, mode='bilinear')
        rgb_down16 = F.interpolate(rgb_down8, scale_factor=0.5, mode='bilinear')
        #rgb_down32 = F.interpolate(rgb_down16, scale_factor=0.5, mode='bilinear')
        #rgb_up16 = F.interpolate(rgb_down32, rgb_down16.shape[2:], mode='bilinear')
        rgb_up8 = F.interpolate(rgb_down16, rgb_down8.shape[2:], mode='bilinear')
        rgb_up4 = F.interpolate(rgb_down8, rgb_down4.shape[2:], mode='bilinear')
        rgb_up2 = F.interpolate(rgb_down4, rgb_down2.shape[2:], mode='bilinear')
        rgb_up = F.interpolate(rgb_down2, x.shape[2:], mode='bilinear')
        lap1 = x - rgb_up
        lap2 = rgb_down2 - rgb_up2
        lap3 = rgb_down4 - rgb_up4
        lap4 = rgb_down8 - rgb_up8
        #lap5 = rgb_down16 - rgb_up16

        #CSP模块
        x1 = self.inc(x)  # 64 256*256

        # d1 = self.down1(x1) #128
        # d2 = self.down2(d1) #256
        # d3 = self.down3(d2) #512
        # d4 = self.down4(d3) #1024

        en_x1att = self.cbam_64(x1)
        x2 = self.down1(en_x1att) #128 128*128
        # cat_x2 = torch.cat([x2, d1], dim=1) #256
        # cat_x2 = self.conv1(cat_x2) #256->128

        en_x2att = self.cbam_128(x2)
        x3 = self.down2(en_x2att) #256 64*64
        # cat_x3 = torch.cat([x3, d2],dim=1)  # 512
        # cat_x3 = self.conv2(cat_x3)  # 512->256

        en_x3att = self.cbam_256(x3)
        x4 = self.down3(en_x3att) # 512 32*32
        # cat_x4 = torch.cat([x4, d3],dim=1)  # 1024
        # cat_x4 = self.conv3(cat_x4)  # 1024->512

        en_x4att = self.cbam_512(x4)
        x5 = self.down4(en_x4att)  # 1024 16*16
        # cat_x5 = torch.cat([x5, d4],dim=1)  # 2048
        # cat_x5 = self.conv4(cat_x5)  # 2048->1024

        #bridge-aspp
        # x1aspp = self.ASPP_64(x1)
        # x2aspp = self.ASPP_128(x2)
        # x3aspp = self.ASPP_256(x3)
        # x4aspp = self.ASPP_512(x4)

        x1up = self.up1(x5, x4)
        x2up = self.up2(x1up, x3)
        x3up = self.up3(x2up, x2)
        x4up = self.up4(x3up, x1)
        # x1up = self.up1(x5att, x4att, lap4)
        # x2up = self.up2(x1up, x3att, lap3)
        # x3up = self.up3(x2up, x2att, lap2)
        # x4up = self.up4(x3up, x1att, lap1)
        outs = []
        #outs = [x5, x1up, x2up, x3up, x4up]
        dec_outs1 = [x2up, x3up, x4up]
        #outs.append(dec_outs1)
        #dec_outs2 = [x1up, x2up, x3up]
        #outs.append(dec_outs2)
        #logits = self.outc(x4up)
        return dec_outs1

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(AttUNet1, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'
