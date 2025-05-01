'''
This did not result in the best model. Best model came from unet_pbc_prelu with bce and mse loss

splitting output channels because experiments in another project for one of the co-authors revealed better performance when splitting output channels this way.
'''
import torch.nn as nn
import torch
import torch.nn.functional as F

class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConv2d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Pad input tensor with periodic boundary conditions
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x


class UNetPBCPrelu(nn.Module):
    def __init__(self):
        super(UNetPBCPrelu, self).__init__()
        self.blk1 = nn.Sequential(
            PeriodicConv2d(2, 64, 3, 1, 1), #input channels = 2, changed from 3
            # nn.BatchNorm2d(64),
            nn.PReLU(64, 0.02),

            PeriodicConv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.PReLU(64, 0.02),
        )
        self.blk2 = nn.Sequential(
            PeriodicConv2d(64, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.PReLU(128, 0.02),

            PeriodicConv2d(128, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.PReLU(128, 0.02),
        )
        self.blk3 = nn.Sequential(
            PeriodicConv2d(128, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.PReLU(256, 0.02),

            PeriodicConv2d(256, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.PReLU(256, 0.02),
        )
        self.blk4 = nn.Sequential(
            PeriodicConv2d(256, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.PReLU(512, 0.02),

            PeriodicConv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.PReLU(512, 0.02),
        )
        self.blk5 = nn.Sequential(
            PeriodicConv2d(512, 1024, 3, 1, 1),
            # nn.BatchNorm2d(1024),
            nn.PReLU(1024, 0.02),

            PeriodicConv2d(1024, 1024, 3, 1, 1),
            # nn.BatchNorm2d(1024),
            nn.PReLU(1024, 0.02),
        )
        
        self.blkUp1 = nn.Sequential(
            PeriodicConv2d(1024, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.PReLU(512, 0.02),

            PeriodicConv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.PReLU(512, 0.02),
        )
        self.blkUp2 = nn.Sequential(
            PeriodicConv2d(512, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.PReLU(256, 0.02),

            PeriodicConv2d(256, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.PReLU(256, 0.02),
        )
        self.blkUp3 = nn.Sequential(
            PeriodicConv2d(256, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.PReLU(128, 0.02),

            PeriodicConv2d(128, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.PReLU(128, 0.02),
        )
        self.blkUp4 = nn.Sequential(
            PeriodicConv2d(128, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.PReLU(64, 0.02),

            PeriodicConv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.PReLU(64, 0.02),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0)
        )
        # self.lastlayer = nn.ConvTranspose2d(64, 2, 3, 1, 1) #output channels = 2, changed from 1
        self.lastlayer_cell = nn.Sequential(nn.ConvTranspose2d(64,1,3,1,1),nn.PReLU(1,1.0))
        self.lastlayer_vegf = nn.Sequential(nn.ConvTranspose2d(64,1,3,1,1),nn.PReLU(1,0.02))

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        # xfinal = self.lastlayer(x9)
        xfinal_cell = self.lastlayer_cell(x9) 
        xfinal_vegf = self.lastlayer_vegf(x9)
        xfinal = torch.cat((xfinal_cell, xfinal_vegf), dim=1)

        return xfinal
    
class ResUNetPBCPrelu(nn.Module):
    def __init__(self):
        super(ResUNetPBCPrelu, self).__init__()
        self.blk1 = nn.Sequential(
            PeriodicConv2d(2, 64, 3, 1, 1), #input channels = 2, changed from 3
            # nn.BatchNorm2d(64),
            nn.PReLU(64, 0.02),

            PeriodicConv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.PReLU(64, 0.02),
        )
        self.blk2 = nn.Sequential(
            PeriodicConv2d(64, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.PReLU(128, 0.02),

            PeriodicConv2d(128, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.PReLU(128, 0.02),
        )
        self.blk3 = nn.Sequential(
            PeriodicConv2d(128, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.PReLU(256, 0.02),

            PeriodicConv2d(256, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.PReLU(256, 0.02),
        )
        self.blk4 = nn.Sequential(
            PeriodicConv2d(256, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.PReLU(512, 0.02),

            PeriodicConv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.PReLU(512, 0.02),
        )
        self.blk5 = nn.Sequential(
            PeriodicConv2d(512, 1024, 3, 1, 1),
            # nn.BatchNorm2d(1024),
            nn.PReLU(1024, 0.02),

            PeriodicConv2d(1024, 1024, 3, 1, 1),
            # nn.BatchNorm2d(1024),
            nn.PReLU(1024, 0.02),
        )
        
        self.blkUp1 = nn.Sequential(
            PeriodicConv2d(1024, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.PReLU(512, 0.02),

            PeriodicConv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.PReLU(512, 0.02),
        )
        self.blkUp2 = nn.Sequential(
            PeriodicConv2d(512, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.PReLU(256, 0.02),

            PeriodicConv2d(256, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.PReLU(256, 0.02),
        )
        self.blkUp3 = nn.Sequential(
            PeriodicConv2d(256, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.PReLU(128, 0.02),

            PeriodicConv2d(128, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.PReLU(128, 0.02),
        )
        self.blkUp4 = nn.Sequential(
            PeriodicConv2d(128, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.PReLU(64, 0.02),

            PeriodicConv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.PReLU(64, 0.02),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0)
        )
        # self.lastlayer = nn.ConvTranspose2d(64, 2, 3, 1, 1) #output channels = 2, changed from 1
        self.lastlayer_cell = nn.Sequential(nn.ConvTranspose2d(64,1,3,1,1),nn.PReLU(1,1.0))
        self.lastlayer_vegf = nn.Sequential(nn.ConvTranspose2d(64,1,3,1,1),nn.PReLU(1,0.02))

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        # xfinal = self.lastlayer(x9)
        xfinal_cell = self.lastlayer_cell(x9) 
        xfinal_vegf = self.lastlayer_vegf(x9)
        xfinal = torch.cat((xfinal_cell, xfinal_vegf), dim=1)

        xfinal += x # residual connection

        return xfinal