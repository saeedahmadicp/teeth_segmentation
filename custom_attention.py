import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Attention_block

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class spatial_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(spatial_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups=in_channels//8,num_channels=in_channels),
            nn.Conv2d(in_channels, out_channels,  kernel_size=7, stride=1, padding=3, dilation=1),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups=in_channels//8,num_channels=in_channels),
            nn.Conv2d(in_channels, out_channels,  kernel_size=5, stride=1, padding=2, dilation=1),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.GroupNorm(num_groups=in_channels//8,num_channels=in_channels),
            nn.Conv2d(in_channels, out_channels,  kernel_size=3, stride=1, padding=1, dilation=1),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.GroupNorm(num_groups=in_channels//8,num_channels=in_channels),
            nn.Conv2d(in_channels, out_channels,  kernel_size=1, stride=1, padding=0, dilation=1),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        concat = [self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)]
        return torch.cat(concat, dim=1)

class SpatialAttention(nn.Module):
    def __init__(self, in_channel, kernel_size = 7):
        super(SpatialAttention, self).__init__()

        self.x_pool = ChannelPool()
        self.g_pool = ChannelPool()

        self.x_block = spatial_block(in_channels=in_channel, out_channels=1)
        self.g_block = spatial_block(in_channels=in_channel, out_channels=1)

        self.scale_x = nn.Conv2d(6, 1,kernel_size=1, stride=1, padding=0, dilation=4)
        self.scale_g = nn.Conv2d(6, 1,kernel_size=1, stride=1, padding=0, dilation=4)
   
       
    def forward(self, x, g):
        x1 = self.x_pool(x)
        x2 = self.x_block(x)
        x_out = torch.cat((x1, x2), dim=1)

        g1 = self.g_pool(g)
        g2 = self.g_block(g)
        g_out = torch.cat((g1, g2), dim=1)

        scale_x = self.scale_x(x_out)
        scale_g = self.scale_g(g_out)

        output = torch.sigmoid(scale_x+scale_g)
        return x* output 

def test():
    x = torch.randn((3, 64, 256, 256))
    #attention = CBAM(gate_channels=64)
    #output = attention(x)
    output = SpatialAttention(in_channel=64)
    x2 = output(x, x)
    print(x2.shape)
    
    


if __name__ == "__main__":
    test()