
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.05)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upconv = nn.ConvTranspose2d(in_ch * 2/3, in_ch*2/3, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, prevx, x1, x2):
    	prevx = self.upconv(prevx)
    	# input is CHW
    	diffY = x2.size()[2] - prevx.size()[2]
    	diffX = x2.size()[3] - prevx.size()[3]

    	prevx = F.pad(prevx, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))

    	# for padding issues, see 
    	# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    	# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    	x = torch.cat([x2, x1, prevx], dim=1)
    	x = self.conv(x)
    	return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        return x
