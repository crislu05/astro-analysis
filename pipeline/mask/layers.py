"""Named boolean mask layers combined with OR."""

from dataclasses import dataclass, field

import numpy as np

from pipeline.mask.config import LAYER_NAMES


@dataclass
class MaskLayers:
    shape: tuple[int, int]

    border: np.ndarray = field(init=False)
    saturated: np.ndarray = field(init=False)
    bleed: np.ndarray = field(init=False)
    halo: np.ndarray = field(init=False)
    streak: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        for name in LAYER_NAMES:
            object.__setattr__(self, name, np.zeros(self.shape, dtype=bool))

    def combined(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=bool)
        for name in LAYER_NAMES:
            out |= getattr(self, name)
        return out

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in LAYER_NAMES}