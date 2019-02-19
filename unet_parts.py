# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def viewActivations(x, net, logger, windowName = 'x'):

    padding = 1
    ubound = 255.0

    xout = net.forward(x, get_features = True)

    for i in range(len(xout)-2):
        
        xi = xout[i]

        xi = xi.detach().cpu().numpy()

        print(xi.shape)
        print('')

        (N, C, H, W) = xi.shape
        grid_height = H * C + padding * (C - 1)
        grid_width = W * N + padding * (N - 1)

        xgrid = np.zeros((grid_height, grid_width))

        x0, x1 = 0, W
        idx = 0

        while x1 <= grid_width:

            y0, y1 = 0, H
            channel = 0

            while y1 <= grid_height:
                
                ximg = xi[idx, channel, :, :]

                xlow, xhigh = np.min(ximg), np.max(ximg)
                if xhigh != xlow:
                    xgrid[y0:y1, x0:x1] = ubound * (ximg - xlow) / (xhigh - xlow)

                y0 += H + padding
                y1 += H + padding

                channel += 1

            x0 += W + padding
            x1 += W + padding

            idx += 1

        logger.plotImages(images=xgrid, nrow=10, win=windowName + str(i), caption = windowName)


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.05)

def rangeNormalize(tensors, provideRange = False, maxVal = 0, minVal = 0):
    outputs = torch.tensor(([]), device=tensors.get_device())
    for idx, tensor in enumerate(tensors):

    #if user provided max and min values then use them, otherwise get it from the data
        if(~provideRange):
            minVal = tensor.min()
            maxVal = tensor.max()
        else:
            print("Used user supplied values")
        a = 1.0 / ((maxVal - minVal) + 1e-8)
        b = 1.0 - a * maxVal
        outputs = torch.cat((outputs, tensor.mul(a).add(b).unsqueeze(0)), 0)
    return (outputs, maxVal, minVal)

def meanStdNormalize(tensors, targetMean, targetStd):
    imageWidth = tensors.shape[2]
    imageHeight = tensors.shape[3]

    tensors1dView = tensors.view(tensors.shape[0], tensors.shape[1], imageWidth*imageHeight)
    mean = tensors1dView.mean(dim=0, keepdim=True)
    stdDev = tensors1dView.std(dim=0, keepdim=True)

    tensors1dView = (tensors1dView - mean) / (stdDev + 1e-8)

    tensors1dView = tensors1dView * targetStd
    tensors1dView = tensors1dView + targetMean

    mean = tensors1dView.mean(dim=0, keepdim=True)
    stdDev = tensors1dView.std(dim=0, keepdim=True)

    return tensors

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
            ReplicatePad2d((1,1,1,1)),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),

            nn.LeakyReLU(0.1),
            ReplicatePad2d((1,1,1,1)),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1)
            
            #ReplicatePad2d((1,1,1,1)),
            #nn.Conv2d(out_ch, out_ch, 3),
            #nn.BatchNorm2d(out_ch),
            #nn.LeakyReLU(0.1),

            #ReplicatePad2d((1,1,1,1)),
            #nn.Conv2d(out_ch, out_ch, 3),
            #nn.BatchNorm2d(out_ch),
            #nn.LeakyReLU(0.1)
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
            self.upconv = nn.ConvTranspose2d(in_ch//2, in_ch//2, 3, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))

        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        #self.pad = ReplicatePad2d((1,1,1,1))
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv.apply(init_weights)
        #self.batchNorm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        #x = self.pad(x)
        x = self.conv(x)
        #x = self.batchNorm(x)
        return x
