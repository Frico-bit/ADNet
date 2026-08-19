import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    """Standard residual block for decoder."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()

        self.projection = (in_channels != out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding='same', bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                              stride=1, padding='same', bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.projection:
            identity = self.bn3(self.project(identity))
        
        out += identity
        out = F.relu(out)
        
        return out

class BaseUNet(nn.Module):
    """Base UNet structure (identical to baseline models)."""
    def __init__(self, in_channels=3,   
                 out_channels=1, input_conv=False, deeper=False,
                 d_channels=None, up_channels=None):
        super().__init__()

        self.deeper = deeper
        
        if d_channels is None: 
            d_channels = [16, 32, 64, 128, 256, 512]
        if up_channels is None:
            up_channels = [32, 64, 128, 256, 512]
        self.d_channels = d_channels
        self.up_channels = up_channels
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, d_channels[0], kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(d_channels[0]),
            nn.ReLU(inplace=True)) if input_conv else nn.Identity()
        
        # Decoder blocks (shared)
        self.d_0 = ResidualBlock(d_channels[5], d_channels[4], 3) # 512, 256
        self.d_1 = ResidualBlock(d_channels[4], d_channels[3], 3) # 256, 128
        self.d_2 = ResidualBlock(d_channels[3], d_channels[2], 3) # 128, 64
        self.d_3 = ResidualBlock(d_channels[2], d_channels[1], 3) # 64, 32

        if deeper:
            self.d_4 = ResidualBlock(d_channels[6], d_channels[5], 3)
            self.pool_4 = nn.MaxPool2d(2, 2)
            self.upsample_5 = nn.ConvTranspose2d(up_channels[5], up_channels[4], 2, 2)

        # Pooling layers
        self.pool_0 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    nn.Conv2d(d_channels[1], d_channels[2], kernel_size=1)) # nn.MaxPool2d(2, 2)
        self.pool_1 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    nn.Conv2d(d_channels[2], d_channels[3], kernel_size=1)) # nn.MaxPool2d(2, 2)
        self.pool_2 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    nn.Conv2d(d_channels[3], d_channels[4], kernel_size=1)) # nn.MaxPool2d(2, 2)
        self.pool_3 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    nn.Conv2d(d_channels[4], d_channels[5], kernel_size=1)) # nn.MaxPool2d(2, 2)

        # Upsampling layers
        self.upsample_0 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                        nn.Conv2d(up_channels[4], up_channels[3], kernel_size=1, bias=False))
                                        # nn.BatchNorm2d(up_channels[3]),
                                        # nn.ReLU(inplace=True)) # 512, 256
        self.upsample_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                        nn.Conv2d(up_channels[3], up_channels[2], kernel_size=1, bias=False))
                                        # nn.BatchNorm2d(up_channels[2]),
                                        # nn.ReLU(inplace=True)) # 256, 128
        self.upsample_2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                        nn.Conv2d(up_channels[2], up_channels[1], kernel_size=1, bias=False))
                                        # nn.BatchNorm2d(up_channels[1]),
                                        # nn.ReLU(inplace=True)) # 128, 64
        self.upsample_3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                        nn.Conv2d(up_channels[1], up_channels[0], kernel_size=1, bias=False))
                                        # nn.BatchNorm2d(up_channels[0]),
                                        # nn.ReLU(inplace=True)) # 64, 32
        # self.upsample_0 = UpsamplePixelShuffle(up_channels[4], up_channels[3])
                                        
        # self.upsample_1 = UpsamplePixelShuffle(up_channels[3], up_channels[2])
                                    
        # self.upsample_2 = UpsamplePixelShuffle(up_channels[2], up_channels[1])
        #                                 # nn.BatchNorm2d(up_channels[1]),
        #                                 # nn.ReLU(inplace=True)) # 128, 64
        # self.upsample_3 = UpsamplePixelShuffle(up_channels[1], up_channels[0])
        # Output layer
        self.conv_out = nn.Conv2d(d_channels[1], out_channels, kernel_size=1) # 32, 1
    
    
    def forward(self, input_image):
        
        # input Conv
        x = self.in_conv(input_image)
        
        # Encoder
        x = self.e_0(x)
        skip_0 = x
        x = self.pool_0(x)

        x = self.e_1(x)
        skip_1 = x
        x = self.pool_1(x)

        x = self.e_2(x)
        skip_2 = x
        x = self.pool_2(x)

        x = self.e_3(x)
        skip_3 = x
        x = self.pool_3(x)

        # deeper layer
        if self.deeper:
            x = self.e_4(x)
            skip_4 = x
            x = self.pool_4(x)

        # Bottleneck
        x = self.bottleneck(x)

        if self.deeper:
            x = self.upsample_5(x)
            x = torch.cat([x, skip_4], dim=1)
            x = self.d_4(x)

        # Decoder
        x = self.upsample_0(x)
        x = torch.cat([x, skip_3], dim=1)
        x = self.d_0(x)

        x = self.upsample_1(x)
        x = torch.cat([x, skip_2], dim=1)
        x = self.d_1(x)

        x = self.upsample_2(x)
        x = torch.cat([x, skip_1], dim=1)
        x = self.d_2(x)

        x = self.upsample_3(x)
        x = torch.cat([x, skip_0], dim=1)
        x = self.d_3(x)

        # Output
        out = self.conv_out(x)
        
        return out