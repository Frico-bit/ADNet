import torch
from torch import nn

class SelectiveKernel(nn.Module):
    def __init__(self, ch, kernels=[3,7], reduction=4):
        super().__init__()
        # mid_ch = max(ch // reduction, 32)
        self.kernels = kernels
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=k, padding=k//2, groups=ch, bias=False), 
            # nn.Conv2d(ch, ch, kernel_size=1, bias=False), 
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)) for k in kernels])
        
        self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, ch//reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch//reduction, len(self.kernels)*ch, 1, bias=False)
            )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, cond=None):
        fs = []
        b, c, _, _ = x.size()

        for i in range(len(self.kernels)):
            fs.append(self.convs[i](x))

        feats = torch.stack(fs, dim=1)   # [B, len(Kernels), C, H, W]
        
        if cond is None:
            z = sum(fs)
        else:
            z = cond
        
        a = self.fc(z)                          # [B, len(Kernels)*C, 1, 1]
        a = a.view(b, len(self.kernels), c, 1, 1)
        a = self.softmax(a)
        out = (feats * a).sum(dim=1)
        return out 

class CSKA(nn.Module):
    def __init__(self, ch, kernels=(3,7)):
        super().__init__()
        self.register_buffer("attn", None)
        self.scaleS = nn.Sequential(nn.Conv2d(ch, ch//2, kernel_size=1, bias=False), 
                                   nn.BatchNorm2d(ch//2), nn.ReLU(inplace=True))
        self.scaleD = nn.Sequential(nn.Conv2d(ch, ch//2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(ch//2), nn.ReLU(inplace=True))
        
        self.kernel_selection_d = SelectiveKernel(ch//2, kernels=kernels) # SelectiveKernel(ch, kernels=kernels)
        self.kernel_selection_s = SelectiveKernel(ch//2, kernels=kernels)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch, ch*2, 1, bias=False),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(inplace=True))
        
    def forward(self, dec, skip):
        B, C, H, W = skip.shape
        # ch_attn = self.channel_att(dec)
        # F.avg_pool2d(S0, kernel_size=3, stride=1, padding=1) invece di self.avg
        hf_feats = self.scaleS(skip) # (1+ch_attn) * skip #skip # - self.avg(skip)
        lf_feats = self.scaleD(dec)
        
        skl = self.kernel_selection_d(lf_feats, hf_feats)
        skh = self.kernel_selection_s(hf_feats, lf_feats)

        out = self.fuse(torch.concat([skl+lf_feats, skh+hf_feats], dim=1)) # self.fuse()
        # print(out.shape)
        self.attn = out.mean(dim=1, keepdim=True).detach().cpu()
        return out 