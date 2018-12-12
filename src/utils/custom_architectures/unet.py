# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


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
    def __init__(self, in_ch, out_ch, drop_rate, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.drop_rate = drop_rate

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = F.dropout(x, self.drop_rate)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, scale=32, n_channels=1, n_classes=1, drop_rate=0.5):
        super(UNet, self).__init__()
                
        # Unet part for heatmap
        self.inc = inconv(n_channels, scale)
        self.down1 = down(scale, scale*2)
        self.down2 = down(scale*2, scale*4)
        self.down3 = down(scale*4, scale*8)
        self.down4 = down(scale*8, scale*8)
        self.up1 = up(scale*16, scale*4, drop_rate)
        self.up2 = up(scale*8, scale*2, drop_rate)
        self.up3 = up(scale*4, scale, drop_rate)
        self.up4 = up(scale*2, scale, drop_rate)
        self.outc = outconv(scale, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # initial convolution
        x1 = self.inc(x)
        # downscale
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # deconv
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        count = torch.sum(self.sigmoid(x).view(x.size(0), -1), 1)

        # return output dict
        return {'count': count.detach(),
                'heatmap': x}
