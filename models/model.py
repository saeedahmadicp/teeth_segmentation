import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchviz import make_dot

from .modules import DoubleConv, Attention_block, InceptionBlock, ResNetBlock, DoubleConv_GN
from ..attentions import CBAM
from ..attentions import SpatialAttention

__all__ = ['UNET', 'UNET_GN', 'Attention_UNET', 'CustomAttention_UNET', 'Inception_UNET', 'Inception_Attention_UNET', 'ResUNET', 'ResUNETPlus',
           'ResUNET_with_CBAM', 'ResUNET_with_GN']

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
            ### up convolution
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )

            ### double convolution
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

class UNET_GN(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET_GN, self).__init__()
        #module list for encoder layers
        self.downs = nn.ModuleList()
        #max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #module list for the decoder layers
        self.ups = nn.ModuleList()

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv_GN(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            ### up convolution
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )

            ### double convolution
            self.ups.append(DoubleConv_GN(feature*2, feature))

        self.bottleneck = DoubleConv_GN(features[-1], features[-1]*2)
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


class Attention_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(Attention_UNET, self).__init__()
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
            ### up convolution
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )

            ### attention module
            self.ups.append(
                Attention_block(F_g=feature, F_l=feature, F_int=feature//2)
            )

            ### double convolution
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

        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//3]
            
            #if the resolution of the input image is not completely devisible, then it will skip the reminder
            # and the resolution will not be equal in this case, so we are resizing it incase in they are not equal
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])


            #Attention module 
            Attention_output = self.ups[idx+1](g= x , x=skip_connection)

            #concatenating the skip connections with x
            concat_skip = torch.cat((Attention_output, x), dim=1)

            #passing the concatenated ouptut, to the double convolutional layers
            x = self.ups[idx+2](concat_skip)

        return self.final_conv(x)


class CustomAttention_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(CustomAttention_UNET, self).__init__()
        #module list for encoder layers
        self.downs = nn.ModuleList()
        #max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #module list for the decoder layers
        self.ups = nn.ModuleList()

         # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv_GN(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            ### up convolution
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )

            ### attention module
            self.ups.append(
                SpatialAttention(in_channel=feature)
            )

             ### attention module
            self.ups.append(
                Attention_block(F_g=feature, F_l=feature, F_int=feature//2)
            )

            ### double convolution
            self.ups.append(DoubleConv_GN(feature*2, feature))

        self.bottleneck = DoubleConv_GN(features[-1], features[-1]*2)
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

        for idx in range(0, len(self.ups), 4):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//4]
            
            #if the resolution of the input image is not completely devisible, then it will skip the reminder
            # and the resolution will not be equal in this case, so we are resizing it incase in they are not equal
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])


            #Attention module 
            spatial_output = self.ups[idx+1](g= x , x=skip_connection)

            #attention unet 
           # Attention_output = self.ups[idx+2](g= x ,  x=skip_connection)#x=spatial_output)

            ###
            #combined_attention = spatial_output * Attention_output

            #concatenating the skip connections with x
            concat_skip = torch.cat((spatial_output, x), dim=1)

            

            #passing the concatenated ouptut, to the double convolutional layers
            x = self.ups[idx+3](concat_skip)

        return self.final_conv(x)


class Inception_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(Inception_UNET, self).__init__()
        #module list for encoder layers
        self.downs = nn.ModuleList()
        #max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #module list for the decoder layers
        self.ups = nn.ModuleList()

        #module list for inpception blocks
        self.inception_blocks = nn.ModuleList()

        #up convolution for inception block
        self.ups_inception_block = nn.ModuleList()


        #inception blocks
        for index in  range(0, len(features)):
            if features[index] == features[-1]:
                out_ch = features[index]//4
            else:
                out_ch = features[index+1]//4
            self.inception_blocks.append(InceptionBlock(features[index], out_ch))
            self.ups_inception_block.append(nn.ConvTranspose2d(out_ch*4, out_ch*4, kernel_size=2, stride=2))


        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature, feature2 in zip(reversed(features), [1536,1024, 512, 256]):
            ### up convolution
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )

            ### double convolution
            self.ups.append(DoubleConv(feature2, feature))

            

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

       

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        

        #inception blocks
        output_blocks = []
        for index in range(0, len(self.inception_blocks)):
            if index == 0:
                block = skip_connections[0]
            
            block = self.inception_blocks[index](block)
            output_blocks.append(block)

        #up convolution for the inception block
        up_convolved_block = []
        for index, upconv in enumerate(self.ups_inception_block):
            up_convolved_block.append(upconv(output_blocks[index]))


        #reversing the list of the skip connections
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            #if the resolution of the input image is not completely devisible, then it will skip the reminder
            # and the resolution will not be equal in this case, so we are resizing it incase in they are not equal
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])


            concat_skip = torch.cat((skip_connection,up_convolved_block[(-(idx//2 + 1))], x), dim=1)

            #passing the concatenated ouptut, to the double convolutional layers
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)



class Inception_Attention_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(Inception_Attention_UNET, self).__init__()
        #module list for encoder layers
        self.downs = nn.ModuleList()
        #max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #module list for the decoder layers
        self.ups = nn.ModuleList()

        #module list for inpception blocks
        self.inception_blocks = nn.ModuleList()

        #up convolution for inception block
        self.ups_inception_block = nn.ModuleList()


        #inception blocks
        for index in  range(0, len(features)):
            if features[index] == features[-1]:
                out_ch = features[index]//4
            else:
                out_ch = features[index+1]//4
            self.inception_blocks.append(InceptionBlock(features[index], out_ch))
            self.ups_inception_block.append(nn.ConvTranspose2d(out_ch*4, out_ch*4, kernel_size=2, stride=2))


        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature, feature2 in zip(reversed(features), [1536,1024, 512, 256]):
            ### up convolution
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )

             ### attention module
            self.ups.append(
                Attention_block(F_g=feature, F_l=feature, F_int=feature//2)
            )

            ### double convolution
            self.ups.append(DoubleConv(feature2, feature))

            

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

       

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        

        #inception blocks
        output_blocks = []
        for index in range(0, len(self.inception_blocks)):
            if index == 0:
                block = skip_connections[0]
            
            block = self.inception_blocks[index](block)
            output_blocks.append(block)

        #up convolution for the inception block
        up_convolved_block = []
        for index, upconv in enumerate(self.ups_inception_block):
            up_convolved_block.append(upconv(output_blocks[index]))


        #reversing the list of the skip connections
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//3]
            
            #if the resolution of the input image is not completely devisible, then it will skip the reminder
            # and the resolution will not be equal in this case, so we are resizing it incase in they are not equal
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            #Attention module 
            Attention_output = self.ups[idx+1](g=x, x=skip_connection)

            concat_skip = torch.cat((Attention_output,up_convolved_block[(-(idx//3 + 1))], x), dim=1)

            #passing the concatenated ouptut, to the double convolutional layers
            x = self.ups[idx+2](concat_skip)

        return self.final_conv(x)



class ResUNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=[64, 128, 256, 512]):
        super(ResUNET, self).__init__()

        ## input and encoder blocks
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResNetBlock(filters[0], filters[1], stride=2, padding=1)
        self.residual_conv_2 = ResNetBlock(filters[1], filters[2], stride=2, padding=1)


        ## bridge
        self.bridge = ResNetBlock(filters[2], filters[3], stride=2, padding=1)

        ## decoder blocks
        self.upsample_1 = nn.ConvTranspose2d(filters[3],filters[3], kernel_size=2, stride=2)
        self.up_residual_conv_1 = ResNetBlock(filters[3]+filters[2],filters[2], stride=1, padding=1)

        self.upsample_2 = nn.ConvTranspose2d(filters[2],filters[2], kernel_size=2, stride=2)
        self.up_residual_conv_2 = ResNetBlock(filters[2]+filters[1],filters[1], stride=1, padding=1)

        self.upsample_3 = nn.ConvTranspose2d(filters[1],filters[1], kernel_size=2, stride=2)
        self.up_residual_conv_3 = ResNetBlock(filters[1]+filters[0],filters[0], stride=1, padding=1)

        ## output layer
        self.output_layer = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1,)

    def forward(self, x):
        
        ## Encoder 
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        ## Bridge
        x4 = self.bridge(x3)

        ## Decoder
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv_1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv_2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_residual_conv_3(x9)

        output = self.output_layer(x10)

        return output


class ResUNETPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=[64, 128, 256, 512, 1024]):
        super(ResUNETPlus, self).__init__()

        ## input and encoder blocks
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResNetBlock(filters[0], filters[1], stride=2, padding=1)
        self.residual_conv_2 = ResNetBlock(filters[1], filters[2], stride=2, padding=1)
        self.residual_conv_3 = ResNetBlock(filters[2], filters[3], stride=2, padding=1)


        ## bridge
        self.bridge = ResNetBlock(filters[3], filters[4], stride=2, padding=1)

        ## decoder blocks
        self.upsample_1 = nn.ConvTranspose2d(filters[4],filters[4], kernel_size=2, stride=2)
        self.up_residual_conv_1 = ResNetBlock(filters[4]+filters[3],filters[3], stride=1, padding=1)

        self.upsample_2 = nn.ConvTranspose2d(filters[3],filters[3], kernel_size=2, stride=2)
        self.up_residual_conv_2 = ResNetBlock(filters[3]+filters[2],filters[2], stride=1, padding=1)

        self.upsample_3 = nn.ConvTranspose2d(filters[2],filters[2], kernel_size=2, stride=2)
        self.up_residual_conv_3 = ResNetBlock(filters[2]+filters[1],filters[1], stride=1, padding=1)

        self.upsample_4 = nn.ConvTranspose2d(filters[1],filters[1], kernel_size=2, stride=2)
        self.up_residual_conv_4 = ResNetBlock(filters[1]+filters[0],filters[0], stride=1, padding=1)

        ## output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1,),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        
        ## Encoder 
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)

        ## Bridge
        x5 = self.bridge(x4)

        ## Decoder
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)
        x7 = self.up_residual_conv_1(x6)

        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)
        x9 = self.up_residual_conv_2(x8)

        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)
        x11 = self.up_residual_conv_3(x10)

        x11 = self.upsample_4(x11)
        x12 = torch.cat([x11, x1], dim=1)
        x13 = self.up_residual_conv_4(x12)

        output = self.output_layer(x13)

        return output



class ResUNET_with_CBAM(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=[64, 128, 256, 512]):
        super(ResUNET_with_CBAM, self).__init__()

        ## input and encoder blocks
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )


        self.residual_conv_1 = ResNetBlock(filters[0], filters[1], stride=2, padding=1)
        self.residual_conv_2 = ResNetBlock(filters[1], filters[2], stride=2, padding=1)


        ## bridge
        self.bridge = ResNetBlock(filters[2], filters[3], stride=2, padding=1)

        ## decoder blocks
        self.upsample_1 = nn.ConvTranspose2d(filters[3],filters[3], kernel_size=2, stride=2)
        self.up_residual_conv_1 = ResNetBlock(filters[3]+filters[2],filters[2], stride=1, padding=1)
        self.cbam_1 = CBAM(gate_channels=filters[3]+filters[2])

        

        self.upsample_2 = nn.ConvTranspose2d(filters[2],filters[2], kernel_size=2, stride=2)
        self.up_residual_conv_2 = ResNetBlock(filters[2]+filters[1],filters[1], stride=1, padding=1)
        self.cbam_2 = CBAM(gate_channels=filters[2]+filters[1])

        self.upsample_3 = nn.ConvTranspose2d(filters[1],filters[1], kernel_size=2, stride=2)
        self.up_residual_conv_3 = ResNetBlock(filters[1]+filters[0],filters[0], stride=1, padding=1)
        self.cbam_3 = CBAM(gate_channels=filters[1]+filters[0])

        ## output layer
        self.output_layer = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1,)

    def forward(self, x):
        
        ## Encoder 
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        ## Bridge
        x4 = self.bridge(x3)

        ## Decoder
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x5 = self.cbam_1(x5)
        x6 = self.up_residual_conv_1(x5)
        

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x7 = self.cbam_2(x7)
        x8 = self.up_residual_conv_2(x7)
        

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x9 = self.cbam_3(x9)
        x10 = self.up_residual_conv_3(x9)
        

        output = self.output_layer(x10)

        return output


class ResUNET_with_GN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=[64, 128, 256, 512]):
        super(ResUNET_with_GN, self).__init__()

        ## input and encoder blocks
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=filters[0]//8,num_channels=filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResNetBlock(filters[0], filters[1], stride=2, padding=1)
        self.residual_conv_2 = ResNetBlock(filters[1], filters[2], stride=2, padding=1)


        ## bridge
        self.bridge = ResNetBlock(filters[2], filters[3], stride=2, padding=1)

        ## decoder blocks
        self.upsample_1 = nn.ConvTranspose2d(filters[3],filters[3], kernel_size=2, stride=2)
        self.up_residual_conv_1 = ResNetBlock(filters[3]+filters[2],filters[2], stride=1, padding=1)

        self.upsample_2 = nn.ConvTranspose2d(filters[2],filters[2], kernel_size=2, stride=2)
        self.up_residual_conv_2 = ResNetBlock(filters[2]+filters[1],filters[1], stride=1, padding=1)

        self.upsample_3 = nn.ConvTranspose2d(filters[1],filters[1], kernel_size=2, stride=2)
        self.up_residual_conv_3 = ResNetBlock(filters[1]+filters[0],filters[0], stride=1, padding=1)

        ## output layer
        self.output_layer = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1,)

    def forward(self, x):
        
        ## Encoder 
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        ## Bridge
        x4 = self.bridge(x3)

        ## Decoder
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv_1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv_2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_residual_conv_3(x9)

        output = self.output_layer(x10)

        return output



def test():
    x = torch.randn((3, 1, 160, 160))
    
    model = ResUNETPlus(in_channels=1, out_channels=1)
    print(model)
    preds = model(x)
    print(preds.shape,   x.shape)
    #make_dot(preds, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    #assert preds.shape == x.shape



if __name__ == "__main__":
    test()