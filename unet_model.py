# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        #self.down4 = down(512, 512)
        #self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)
        self.outc = outconv(32, n_classes)
        #self.batchNorm = nn.BatchNorm2d(n_classes)
        
        #self.instanceNorm = nn.InstanceNorm2d(1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        #x = self.instanceNorm(x)
        #x = self.batchNorm(x)
        x = meanStdNormalize(x, targetMean=0.0, targetStd=1)#around sqrt(2)
        (x, _, _) = rangeNormalize(x, provideRange=False)
        #x = torch.sigmoid(x)
        return x

class ShallowNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ShallowNet, self).__init__()
        self.convLayer = double_conv(n_channels, n_classes)

    def forward(self, x):
        x = self.convLayer(x)
        x = meanStdNormalize(x, targetMean=0.0, targetStd=1)
        (x, _, _) = rangeNormalize(x, provideRange=False)
        return x