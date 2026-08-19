import torch
from torch import nn

class RSM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.refinement_l = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

        self.refinement_h = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.discovery_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        refinement_h = self.refinement_h(x)
        refinement_l = self.refinement_l(x)

        refined_guidance = abs(refinement_h - refinement_l)
        
        discovered = self.discovery_net(refined_guidance)
        integrated = discovered + x

        return integrated
