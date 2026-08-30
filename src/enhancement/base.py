"""
Base Enhancer Abstract Class for SIH 2026 Video Enhancement Pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import torch


class BaseEnhancer(ABC):
    """
    Abstract base class that all CCTV enhancement models (Zero-DCE, Real-ESRGAN, BasicVSR, RVRT) inherit from.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model: Optional[torch.nn.Module] = None
        self.is_loaded: bool = False

    @abstractmethod
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a single BGR image/frame (numpy array, uint8 [0, 255]).
        Returns the enhanced BGR frame (numpy array, uint8 [0, 255]).
        """
        pass

    def enhance_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhance a list of BGR frames. Default implementation loops over enhance().
        Subclasses can override for batched GPU tensor inference.
        """
        return [self.enhance(f) for f in frames]

    @abstractmethod
    def load_weights(self, weights_path: str):
        """
        Load pre-trained model weights from disk.
        """
        pass

    def to(self, device: str):
        """Move underlying PyTorch model to target device."""
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self
