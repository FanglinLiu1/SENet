# ------------------------------------------------------------------------------------------------
# Integrated AutoEncoder
# Copyright (c) 2024 Fanglin Liu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['FPN', 'BiFPN', 'FPNet']

from thop import profile
from torchinfo import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(0, 1)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPNet(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(FPNet, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv2 = ConvBlock(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv3 = ConvBlock(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv4 = ConvBlock(64, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        # x = torch.cat([c1, c2, c3, c4], dim=3)
        x = c1 + c2 + c3 + c4
        return x


class FPN(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(FPN, self).__init__()

        self.conv1 = ConvBlock(in_channels, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv2 = ConvBlock(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv3 = ConvBlock(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv4 = ConvBlock(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))

        self.lateral1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.lateral2 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.lateral3 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.lateral4 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=False, padding=0)

        self.output1 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.output2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.output3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.output4 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        p4 = self.lateral1(c4)
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p2 = self.lateral3(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        p1 = self.lateral4(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest')

        out1 = self.output1(p1)
        out2 = self.output2(p2)
        out3 = self.output3(p3)
        out4 = self.output4(p4)

        # out = torch.cat([out1, out2, out3, out4], dim=3)
        out = out1 + out2 + out3 + out4
        return out


class BiFPN(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(BiFPN, self).__init__()

        self.conv1 = ConvBlock(in_channels, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv2 = ConvBlock(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv3 = ConvBlock(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv4 = ConvBlock(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))

        self.lateral1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.lateral2 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.lateral3 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.lateral4 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=False, padding=0)

        self.upsample1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.upsample2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.upsample3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False, padding=0)

        self.output1 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.output2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.output3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.output4 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        p4 = self.lateral1(c4)
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p2 = self.lateral3(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        p1 = self.lateral4(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest')

        u1 = self.upsample1(p1) + F.interpolate(p2, size=p1.shape[2:], mode='nearest')
        u2 = self.upsample2(p2) + F.interpolate(p3, size=p2.shape[2:], mode='nearest')
        u3 = self.upsample3(p3) + F.interpolate(p4, size=p3.shape[2:], mode='nearest')

        out1 = self.output1(u1)
        out2 = self.output2(u2)
        out3 = self.output3(u3)
        out4 = self.output4(p4)

        # out = torch.cat([out1, out2, out3, out4], dim=3)
        out = out1 + out2 + out3 + out4
        return out


if __name__ == "__main__":
    batch_size = 128
    in_channels = 1
    in_height, in_width = 2, 128
    out_channels = 64

    # model = FPNet(in_channels, out_channels)
    model = FPN(in_channels, out_channels)
    # model = BiFPN(in_channels, out_channels)
    x = torch.randn(batch_size, in_channels, in_height, in_width)
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Parameters: {params / 1e6:.2f} M")
    summary(model, input_size=(batch_size, x.shape[1], x.shape[2], x.shape[3]))
