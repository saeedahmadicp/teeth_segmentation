##########################################
# @subject : Unet++ implementation       #
# @author  : perryxin                    #
# @date    : 2018.12.27                  #
##########################################
## pytorch implementation of unet++ , just use its main idea, the model is not the same as the origin unet++ mentioned in paper
## paper : UNet++: A Nested U-Net Architecture for Medical Image Segmentation
## https://arxiv.org/abs/1807.10165
import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv2(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , out_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(out_ch, out_ch, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

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
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2):  # x1--up , x2 ---down
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (
            diffY // 2, diffY - diffY // 2,
            diffX // 2, diffX - diffX // 2,))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up3, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3):
        # print(x1.shape)
        x1 = self.up(x1)
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class up4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up4, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3, x4):  # x1--up , x2 ---down
        # print(x1.shape)
        x1 = self.up(x1)
        x = torch.cat([x4, x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = F.sigmoid(x)
        return x


class double_conv(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_in(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv_in, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 5, padding=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv_in(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


cc = 16  # you can change it to 8, then the model can be more faster ,reaching 35 fps on cpu when testing


class Unet_2D(nn.Module):
    def __init__(self, n_channels, n_classes, mode='train'):
        super(Unet_2D, self).__init__()
        self.inconv = inconv(n_channels, cc)
        self.down1 = down(cc, 2 * cc)
        self.down2 = down(2 * cc, 4 * cc)
        self.down3 = down(4 * cc, 8 * cc)
        self.up1 = up(12 * cc, 4 * cc)
        self.up20 = up(6 * cc, 2 * cc)
        self.up2 = up3(8 * cc, 2 * cc)
        self.up30 = up(3 * cc, cc)
        self.up31 = up3(4 * cc, cc)
        self.up3 = up4(5 * cc, cc)
        self.outconv = outconv(cc, n_classes)
        self.mode = mode

    def forward(self, x):
        if self.mode == 'train':  # use the whole model when training
            x1 = self.inconv(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
            x21 = self.up20(x3, x2)
            x = self.up2(x, x21, x2)
            x11 = self.up30(x2, x1)
            x12 = self.up31(x21, x11, x1)
            x = self.up3(x, x12, x11, x1)
            #output 0 1 2
            y2 = self.outconv(x)
            y0 = self.outconv(x11)
            y1 = self.outconv(x12)
            return y0, y1, y2
        else:  # prune the model when testing
            x1 = self.inconv(x)
            x2 = self.down1(x1)
            x11 = self.up30(x2, x1)
            # output 0
            y0 = self.outconv(x11)
            return y0


if __name__ == '__main__':
    import time

    x = torch.rand((1, 3, 256, 256))
    lnet = Unet_2D(3, 1, 'test')
    # calculate model size
    print('    Total params: %.2fMB' % (sum(p.numel() for p in lnet.parameters()) / (1024.0 * 1024) * 4))
    t1 = time.time()
    ##test for its speed on cpu
    for i in range(60):
        y0 = lnet(x)
    t2 = time.time()
    print('fps: ', 60 / (t2 - t1))
    print(y0.shape)