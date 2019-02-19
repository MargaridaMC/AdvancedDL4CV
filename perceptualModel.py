# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from perceptualParts import *

def correct_size(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2), 'replicate')
    return x1

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.inc = inconv(1, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 128)
        #self.down4 = down(128, 128)
        #self.fcDown1 = nn.Linear(64*128, 2*128) #assuming the input to the network was of size 64x64
        #self.fcUp1 = nn.Linear(2*128, 64*128)
        #self.up1 = up(128, 128)
        self.up2 = up(128, 128)
        self.up3 = up(128, 64)
        self.up4 = up(64, 32)
        self.outc = outconv(32, 1)

    def forward(self, x, get_features = False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x4 = x4.view((x4.size(0), 128*64))
        #linearFeatures = self.fcDown1(x4)

        if get_features:
            return (x1, x2, x3, x4)#, linearFeatures)

        #x = self.fcUp1(linearFeatures)
        #x = x.view((x4.size(0), 128, 8, 8))
        #x = self.up1(x5)
        #x = correct_size(x, x4)
        x = self.up2(x4)
        x = correct_size(x, x3)
        x = self.up3(x)
        x = correct_size(x, x2)
        x = self.up4(x)
        x = correct_size(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)