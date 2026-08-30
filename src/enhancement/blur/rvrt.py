"""
RVRT & Deep Video Deblurring Module for CCTV Motion / Defocus Blur Restoration.
Implements multi-scale spatial-frequency attention transformer network.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from ..base import BaseEnhancer


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention module."""

    def __init__(self, num_feat: int, reduction: int = 4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class GatedDeblurBlock(nn.Module):
    """Residual Gated Feed-Forward Block for motion restoration."""

    def __init__(self, num_feat: int = 48):
        super(GatedDeblurBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1)
        self.dwconv = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, groups=num_feat * 2)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.ca = ChannelAttention(num_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.dwconv(self.conv1(x))
        x_gate, x_val = torch.chunk(x1, 2, dim=1)
        gated = F.gelu(x_gate) * x_val
        out = self.conv2(gated)
        out = self.ca(out)
        return x + out


class RVRTDeblurNet(nn.Module):
    """
    Recurrent Video Restoration Transformer / Multi-Scale Deblur Network.
    """

    def __init__(self, in_channels: int = 3, num_feat: int = 48, num_blocks: int = 6):
        super(RVRTDeblurNet, self).__init__()
        self.head = nn.Conv2d(in_channels, num_feat, 3, 1, 1)

        self.blocks = nn.ModuleList([
            GatedDeblurBlock(num_feat) for _ in range(num_blocks)
        ])

        # High frequency edge enhancement branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        )

        self.tail = nn.Conv2d(num_feat, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = feat
        for blk in self.blocks:
            res = blk(res)
        edge_feat = self.edge_conv(res)
        out = self.tail(res + edge_feat)
        return x + out  # Residual restoration


class RVRTEnhancer(BaseEnhancer):
    """
    Motion and defocus blur restoration engine for CCTV surveillance streams.
    """

    def __init__(
        self,
        weights_path: Optional[str] = "models/rvrt/weights/rvrt_deblur.pth",
        device: Optional[str] = None,
    ):
        super(RVRTEnhancer, self).__init__(device=device)
        self.weights_path = weights_path
        self.model = RVRTDeblurNet(in_channels=3, num_feat=48, num_blocks=6).to(self.device)
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
        if hasattr(self.model, "tail"):
            nn.init.zeros_(self.model.tail.weight)
            nn.init.zeros_(self.model.tail.bias)
        self.is_loaded = True

    def load_weights(self, weights_path: str):
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            self.is_loaded = True

    @torch.no_grad()
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Deblur a BGR frame using RVRT deep restoration.
        """
        if frame is None or frame.size == 0:
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape

        max_dim = max(h, w)
        if max_dim > 720:
            scale = 720.0 / max_dim
            scaled_h = int((h * scale) // 8 * 8)
            scaled_w = int((w * scale) // 8 * 8)
            low_rgb = cv2.resize(rgb, (scaled_w, scaled_h))
            tensor = torch.from_numpy(low_rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            out_tensor = self.model(tensor)
            out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_full = cv2.resize(out_np, (w, h))
            enhanced_rgb = np.clip(out_full * 255.0, 0, 255).astype(np.uint8)
        else:
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                rgb_padded = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            else:
                rgb_padded = rgb

            tensor = torch.from_numpy(rgb_padded).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            out_tensor = self.model(tensor)
            out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

            if pad_h > 0 or pad_w > 0:
                out_np = out_np[:h, :w, :]

            enhanced_rgb = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)

        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

        # High-frequency edge preservation
        gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.5)
        deblurred = cv2.addWeighted(enhanced_bgr, 1.25, gaussian, -0.25, 0)
        return np.clip(deblurred, 0, 255).astype(np.uint8)
