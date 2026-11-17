"""Inputs shared across photometry stages."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pipeline.photometry import config


@dataclass
class PhotometryInputs:
    data: np.ndarray
    mask: np.ndarray
    segm: np.ndarray
    gain: float = config.DEFAULT_GAIN

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        self.mask = np.asarray(self.mask, dtype=bool)
        self.segm = np.asarray(self.segm, dtype=np.int64)
        grown = (
            ndimage.binary_dilation(self.segm > 0, iterations=config.SEG_GROW)
            if config.SEG_GROW > 0
            else (self.segm > 0)
        )
        self.not_sky = self.mask | grown