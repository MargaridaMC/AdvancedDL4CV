# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class ReplicatePad2d(nn.Module):
    def __init__(self, padding):
        super(ReplicatePad2d, self).__init__()
        self.pad = F.pad
        self.padding = padding

    def forward(self, x):
        x = self.pad(x, self.padding, 'replicate')
        return x

class double_conv(nn.Module):
    '''(conv => BN => LeakyReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            ReplicatePad2d((2,2,2,2)),
            nn.Conv2d(in_ch, out_ch, 5),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1),
            ReplicatePad2d((2,2,2,2)),
            nn.Conv2d(out_ch, out_ch, 5),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )
        self.conv.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
                
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too
        if bilinear:
            self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upconv = nn.ConvTranspose2d(in_ch, in_ch, 3, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1):
        x1 = self.upconv(x1)
        x = self.conv(x1)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        return x
