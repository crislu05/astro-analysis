"""Fixed-width edge strip."""

from pipeline.mask import config
from pipeline.mask.layers import MaskLayers


def apply_border(layers: MaskLayers, width: int | None = None) -> None:
    w = config.BORDER_WIDTH if width is None else width
    layers.border[:w, :] = True
    layers.border[-w:, :] = True
    layers.border[:, :w] = True
    layers.border[:, -w:] = True