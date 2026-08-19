from torch import nn
import torch
from torch.nn import functional as F
from model.base import BaseUNet
from model.blocks.cska import CSKA
from model.blocks.adblock import ADBlock
import torchvision.models as models

class ADNet(BaseUNet):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__(in_channels, out_channels, 
                        input_conv=False)
        
        efficientnet = models.efficientnet_v2_s(weights='DEFAULT' if pretrained else None)
        
        self.downsample_block = nn.Sequential(
           efficientnet.features[0],  # Stem: stride=2, 3→24 channels
           efficientnet.features[1], # First Fused-MBConv: 24→24 channels
        )
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.e_0 = ADBlock(32, 32) 
        self.e_1 = ADBlock(64, 64) 
        self.e_2 = ADBlock(128, 128) 
        self.e_3 = ADBlock(256, 256)     
        self.bottleneck = ADBlock(512, 512)    

        self.cska0 = CSKA(256, kernels=(3,5,7)) 
        self.cska1 = CSKA(128, kernels=(3,5,7)) 
        self.cska2 = CSKA(64, kernels=(3,5)) 
        self.cska3 = CSKA(32, kernels=(3,5)) 
   
    def forward(self, input_image):
        original_size = input_image.shape[2:]
        #supervision = []
        # input Conv
        x = self.downsample_block(input_image)
        x = self.in_conv(x)
        
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

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.upsample_0(x)
        x = self.cska0(x, skip_3)
        x = self.d_0(x)
        
        x = self.upsample_1(x)
        x = self.cska1(x, skip_2) 
        x = self.d_1(x)
        
        x = self.upsample_2(x)
        x = self.cska2(x, skip_1) 
        x = self.d_2(x)
        
        x = self.upsample_3(x)
        x = self.cska3(x, skip_0)  
        x = self.d_3(x)
        
        # Output
        out = self.conv_out(x)
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        return out

if __name__ == "__main__":
    model = ADNet()
    x = torch.randn(2, 3, 352, 352)
    with torch.no_grad():
        out = model(x)
        out
    print(sum([l.numel() for l in model.parameters() if l.requires_grad]))