import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchviz import make_dot


class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            #first convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            #2nd convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
        
        
    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        #module list for encoder layers
        self.downs = nn.ModuleList()
        #max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #module list for the decoder layers
        self.ups = nn.ModuleList()

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        #reversing the list of the skip connections
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            #if the resolution of the input image is not completely devisible, then it will skip the reminder
            # and the resolution will not be equal in this case, so we are resizing it incase in they are not equal
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            #concatenating the skip connections with x
            concat_skip = torch.cat((skip_connection, x), dim=1)

            #passing the concatenated ouptut, to the double convolutional layers
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 162, 162))
    model = UNET(in_channels=1, out_channels=1)
    #print(model)
    preds = model(x)
    print(preds.shape,   x.shape)
    #make_dot(preds, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    assert preds.shape == x.shape



if __name__ == "__main__":
    test()