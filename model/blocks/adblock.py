import torch
from torch import nn
from torchvision.ops import DeformConv2d
from .rsm import RSM

class RAGuidedDiscoveryCM(nn.Module):
    def __init__(self, in_channels, latent_dim=64, max_offset=4.0):
        super().__init__()
        self.max_offset = max_offset
        self.latent_dim = latent_dim

        self.rsm = RSM(in_channels)
        
        # 4. Offset & Mask Generators
        # These now take the refined Axial + Boundary guidance
        self.offset_gen = nn.Conv2d(in_channels, latent_dim, 3, padding=1)
        self.mask_gen = nn.Conv2d(in_channels, latent_dim // 2, 3, padding=1)
        
        # 5. Spatial Weighting (SES-inspired)
        # Redistributes weights to focus on complex edges 
        self.ses_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if m in [self.offset_gen, self.mask_gen]:
                nn.init.constant_(m.weight, 0)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        integrated = self.rsm(x)

        spatial_weights = self.ses_layer(integrated) # global attention
        guided_features = integrated * spatial_weights
        
        # F. Generate Outputs
        offset_latents = self.offset_gen(guided_features) # local attention for the offset
        mask_latents = self.mask_gen(guided_features)
        
        return integrated, offset_latents, mask_latents


class ADBlock(nn.Module):
    """
    BaraBlock with Strategy 1: Shared BARA + Lightweight Projections
    
    Key improvements:
    - Single BARA module (shared understanding)
    - Lightweight 1x1 projections per kernel size
    - 79% parameter reduction vs per-kernel BARA
    - Consistent saliency prediction
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], max_offset=4.0):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.max_offset = max_offset
        stem_channels = out_channels // 2
        
        # Channel budget for branches (equal parameter allocation)
        inv_kernels = [1.0 / k for k in kernel_sizes]
        sum_inv = sum(inv_kernels)
        branch_budget = out_channels // 2
        
        self.branch_channels = [
            max(8, int(branch_budget * (inv_k / sum_inv))) 
            for inv_k in inv_kernels
        ]
        
        print(f"[SharedBARABlock] in={in_channels}, out={out_channels}, stem={stem_channels}")
        print(f"  Kernel sizes: {kernel_sizes}, Branch channels: {self.branch_channels}")
        
        # ============================================
        # STEM: Initial feature extraction
        # ============================================
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        latent_dim_o = 32
        latent_dim_m = latent_dim_o // 2

        # ============================================
        # SHARED: Single sophisticated BARA module
        # ============================================
        self.shared_bara = RAGuidedDiscoveryCM(
            in_channels=stem_channels,
            latent_dim= latent_dim_o, 
        )
        
        # ============================================
        # PER-KERNEL: Lightweight projections
        # ============================================
        # These adapt shared offsets/masks to different kernel sizes
        
        # For horizontal kernels (1×k)
        self.offset_proj_1xk = nn.ModuleList()
        self.mask_proj_1xk = nn.ModuleList()
        self.deform_convs_1xk = nn.ModuleList()
        
        # For vertical kernels (k×1)
        self.offset_proj_kx1 = nn.ModuleList()
        self.mask_proj_kx1 = nn.ModuleList()
        self.deform_convs_kx1 = nn.ModuleList()


        
        for i, k in enumerate(kernel_sizes):
            ch = self.branch_channels[i]
            
            # Horizontal (1×k)
            # Project base offsets (18) → k-specific offsets (2*k)
            self.offset_proj_1xk.append(
                nn.Conv2d(latent_dim_o, 2*k, 1, bias=True)  # 18 → 2k (2 coords per point)
            )
            # Project base masks (9) → k-specific masks (k)
            self.mask_proj_1xk.append(
                nn.Conv2d(latent_dim_m, k, 1, bias=True)  # 9 → k
            )
            self.deform_convs_1xk.append(
                DeformConv2d(stem_channels, ch, (1, k), padding=(0, k//2), bias=False)
            )
            
            # Vertical (k×1)
            self.offset_proj_kx1.append(
                nn.Conv2d(latent_dim_o, 2*k, 1, bias=True)
            )
            self.mask_proj_kx1.append(
                nn.Conv2d(latent_dim_m, k, 1, bias=True)
            )
            self.deform_convs_kx1.append(
                DeformConv2d(stem_channels, ch, (k, 1), padding=(k//2, 0), bias=False)
            )
        
        # Initialize projections to approximate identity (start conservative)
        self._init_projections()
        
        # Calculate total channels for fusion
        self.total_channels = stem_channels + (2 * sum(self.branch_channels))

        self.fusion = nn.Sequential(
            nn.Conv2d(self.total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_projections(self):
        """Initialize projection layers conservatively"""
        for module_list in [self.offset_proj_1xk, self.offset_proj_kx1,
                            self.mask_proj_1xk, self.mask_proj_kx1]:
            for proj in module_list:
                # Small random init (don't start at zero - allow learning)
                nn.init.normal_(proj.weight, mean=0.0, std=0.01)
                if proj.bias is not None:
                    nn.init.constant_(proj.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, H, W]
        
        Returns:
            fused: [B, out_channels, H, W] fused features
            saliency_map: [B, 1, H, W] single saliency map (for supervision)
        """
        # Step 1: Stem processing
        x_stem = self.stem(x)
        
        # Step 2: Shared BARA - generates base offsets/masks/saliency
        integrated, base_offsets, base_masks = self.shared_bara(x_stem)
        # base_offsets: [B, 18, H, W] for 3x3
        # base_masks: [B, 9, H, W] for 3x3
        # saliency_map: [B, 1, H, W]
        
        # Step 3: Collect features from all branches
        branch_features = [x_stem]  # Start with stem output
        
        for i, k in enumerate(self.kernel_sizes):
            # === Horizontal branch (1×k) ===
            # Project base guidance to k-specific
            offsets_h = self.offset_proj_1xk[i](base_offsets)
            masks_h = torch.sigmoid(self.mask_proj_1xk[i](base_masks))
            
            # Apply deformable convolution
            feat_h = self.deform_convs_1xk[i](integrated, offsets_h, masks_h)
            branch_features.append(feat_h)
            
            # === Vertical branch (k×1) ===
            offsets_v = self.offset_proj_kx1[i](base_offsets)
            masks_v = torch.sigmoid(self.mask_proj_kx1[i](base_masks))
            
            feat_v = self.deform_convs_kx1[i](integrated, offsets_v, masks_v)
            branch_features.append(feat_v)
        
        # Step 4: Concatenate all branches
        concat = torch.cat(branch_features, dim=1)
        
        # Verify shape
        assert concat.shape[1] == self.total_channels, \
            f"Channel mismatch: got {concat.shape[1]}, expected {self.total_channels}"
        
        fused = self.fusion(concat)
        
        return fused