import torch.nn as nn
import torch 

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(conv_block,self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class encode_layer(nn.Module):
    def __init__(self, channels, first_layer=False):
        super(encode_layer, self).__init__()

        if first_layer:
            self.layer = nn.Sequential(
                conv_block(in_channels=channels[0], out_channels=channels[1]),
                conv_block(in_channels=channels[1], out_channels=channels[1])
            )        

        else:
            self.layer = nn.Sequential(
                nn.MaxPool2d(2),
                conv_block(in_channels=channels[0], out_channels=channels[1]),
                conv_block(in_channels=channels[1], out_channels=channels[1])
            )

    def forward(self,x):
        x = self.layer(x)
        return x 

class decode_layer(nn.Module):
    def __init__(self,channels):
        super(decode_layer,self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels=channels[-1], out_channels=int(channels[-1]/2), kernel_size=2, stride=2)

        self.convolutions = nn.Sequential(
            conv_block(in_channels=channels[1], out_channels=channels[1]),
            conv_block(in_channels=channels[1], out_channels=channels[0])
        )

    def forward(self,x):
        x1, x2 = x
        x1 = self.upconv(x1)
        batch, c, h_x1, w_x1 = x1.shape
        batch, c, h_x2, w_x2 = x2.shape
        x2 = x2[:,:, int(h_x2/2-h_x1/2):int(h_x2/2+h_x1/2), int(w_x2/2-w_x1/2):int(w_x2/2+w_x1/2)]
        x = torch.cat((x2, x1), dim=1)
        x = self.convolutions(x)
        return x

class unet(nn.Module):
    def __init__(self):
        super(unet,self).__init__()

        channels = [3, 64, 128, 256, 512, 1024]

        self.encoder0 = encode_layer(channels[0:2], first_layer=True)
        self.encoder1 = encode_layer(channels[1:3])
        self.encoder2 = encode_layer(channels[2:4])
        self.encoder3 = encode_layer(channels[3:5])
        self.encoder4 = encode_layer(channels[4:])

        self.decoder0 = decode_layer(channels[-2:])
        self.decoder1 = decode_layer(channels[-3:-1])
        self.decoder2 = decode_layer(channels[-4:-2])
        self.decoder3 = decode_layer(channels[-5:-3])

        self.final = nn.Conv2d(in_channels=channels[1], out_channels=4, kernel_size=1)

    def forward(self,x):
        x1 = self.encoder0(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        x6 = self.decoder0([x5, x4])
        x7 = self.decoder1([x6, x3])
        x8 = self.decoder2([x7, x2])
        x9 = self.decoder3([x8, x1])

        x = self.final(x9)

        return x
        
