"""Mutable pipeline state passed between mask stages."""

from dataclasses import dataclass, field

import numpy as np

from pipeline.mask.bleed import BleedRecord
from pipeline.mask.halo import StarMask
from pipeline.mask.layers import MaskLayers
from pipeline.mask.saturated import SaturatedResult


@dataclass
class MaskState:
    data: np.ndarray
    saturation: float
    xx: np.ndarray
    yy: np.ndarray
    layers: MaskLayers
    sky: float = 0.0
    sigma: float = 0.0
    saturated: SaturatedResult | None = None
    bleeds: list[BleedRecord] = field(default_factory=list)
    stars: list[StarMask] = field(default_factory=list)

    @property
    def mask(self) -> np.ndarray:
        return self.layers.combined()