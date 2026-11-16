"""Saturated cores and connected-component metadata."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pipeline.mask import config
from pipeline.mask.layers import MaskLayers


@dataclass
class SaturatedResult:
    labels: np.ndarray
    n_components: int
    slices: list


def detect_saturated(
    data: np.ndarray,
    layers: MaskLayers,
    saturation: float,
) -> SaturatedResult:
    saturated = (data >= config.SATURATION_FRAC * saturation) & ~layers.border
    saturated = ndimage.binary_dilation(saturated, iterations=config.SAT_DILATE_ITER)
    layers.saturated[:] = saturated
    labels, n = ndimage.label(saturated)
    slices = ndimage.find_objects(labels)
    return SaturatedResult(labels=labels, n_components=n, slices=slices)