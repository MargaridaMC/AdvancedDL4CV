# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from unet_parts import *

class Encoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Encoder, self).__init__()
        self.inc = inconv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return[x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Decoder, self).__init__()
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)
        self.outc = outconv(32, n_classes)
        #self.batchNorm = nn.BatchNorm2d(n_classes)
        #self.instanceNorm = nn.InstanceNorm2d(1)

    def forward(self, X):
        [x1, x2, x3, x4, x5] = X
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        #x = self.instanceNorm(x)
        #x = torch.sigmoid(x)
        x = meanStdNormalize(x, targetMean=0.0, targetStd=1)
        #(x, _, _) = rangeNormalize(x, provideRange=False)
        #x = self.batchNorm(x)

        return x
