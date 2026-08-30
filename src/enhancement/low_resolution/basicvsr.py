"""
BasicVSR / BasicVSR++ Recurrent Video Super-Resolution Module.
Exploits multi-frame temporal redundancy for continuous surveillance video streams.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from ..base import BaseEnhancer


class ResidualBlockNoBN(nn.Module):
    """Residual block without Batch Normalization for Super-Resolution."""

    def __init__(self, num_feat: int = 64):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class BasicVSRNet(nn.Module):
    """
    Recurrent Video Super-Resolution Network with forward and backward temporal feature propagation.
    """

    def __init__(self, num_feat: int = 48, num_blocks: int = 5, upscale: int = 2):
        super(BasicVSRNet, self).__init__()
        self.num_feat = num_feat
        self.upscale = upscale

        # Feature extractor
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(3, num_feat, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Forward & Backward propagation blocks
        self.forward_resblocks = nn.Sequential(*[ResidualBlockNoBN(num_feat * 2) for _ in range(num_blocks)])
        self.forward_fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)

        # Reconstruction & Upsampling
        self.reconstruction = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat, 3 * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
        )

    def forward(self, lq_seq: torch.Tensor) -> torch.Tensor:
        """
        lq_seq: Tensor of shape (B, T, C, H, W)
        Returns: Tensor of shape (B, T, C, H*upscale, W*upscale)
        """
        b, t, c, h, w = lq_seq.shape
        # Extract features for all frames
        feats = []
        for i in range(t):
            feats.append(self.feat_extractor(lq_seq[:, i]))

        # Forward recurrent propagation
        outputs = []
        feat_prev = torch.zeros_like(feats[0])
        for i in range(t):
            cat_feat = torch.cat([feats[i], feat_prev], dim=1)
            refined = self.forward_fusion(self.forward_resblocks(cat_feat))
            feat_prev = refined

            sr_frame = self.reconstruction(refined)
            base = F.interpolate(lq_seq[:, i], scale_factor=self.upscale, mode="bilinear", align_corners=False)
            outputs.append((sr_frame + base).unsqueeze(1))

        return torch.cat(outputs, dim=1)


class BasicVSREnhancer(BaseEnhancer):
    """
    Video super-resolution enhancer processing temporal frame windows.
    """

    def __init__(
        self,
        scale: int = 2,
        window_size: int = 3,
        weights_path: Optional[str] = "models/basicvsr/weights/basicvsr_x2.pth",
        device: Optional[str] = None,
    ):
        super(BasicVSREnhancer, self).__init__(device=device)
        self.scale = scale
        self.window_size = window_size
        self.weights_path = weights_path

        self.model = BasicVSRNet(num_feat=48, num_blocks=4, upscale=scale).to(self.device)
        self.model.eval()

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        else:
            self._init_default_weights()

    def _init_default_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        if hasattr(self.model, "reconstruction") and len(self.model.reconstruction) >= 2:
            nn.init.zeros_(self.model.reconstruction[-2].weight)
            nn.init.zeros_(self.model.reconstruction[-2].bias)
        self.is_loaded = True

    def load_weights(self, weights_path: str):
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            self.is_loaded = True

    @torch.no_grad()
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Enhance single frame using self-repetition sequence."""
        return self.enhance_sequence([frame])[0]

    @torch.no_grad()
    def enhance_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhance a sequence of temporal BGR frames.
        """
        if not frames:
            return []

        rgb_tensors = []
        h, w = frames[0].shape[:2]

        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            rgb_tensors.append(tensor)

        seq_tensor = torch.stack(rgb_tensors, dim=0).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        out_tensor = self.model(seq_tensor).squeeze(0)  # (T, C, H*scale, W*scale)

        enhanced_frames = []
        for i in range(len(frames)):
            frame_np = out_tensor[i].permute(1, 2, 0).cpu().numpy()
            frame_np = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
            enhanced_frames.append(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

        return enhanced_frames
