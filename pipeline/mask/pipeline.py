"""Orchestrate mask layers in dependency order."""

from astropy.io.fits import Header

import numpy as np

from pipeline.mask.border import apply_border
from pipeline.mask.bleed import grow_bleed, trace_bleeds
from pipeline.mask.geometry import pixel_grid
from pipeline.mask.halo import apply_halos
from pipeline.mask.layers import MaskLayers
from pipeline.mask.saturated import detect_saturated
from pipeline.mask.sky import fit_sky
from pipeline.mask.state import MaskState


def build_mask_state(data: np.ndarray, header: Header) -> MaskState:
    data = np.asarray(data, dtype=float)
    saturation = float(header["SATURATE"])
    ny, nx = data.shape
    xx, yy = pixel_grid((ny, nx))
    layers = MaskLayers((ny, nx))
    state = MaskState(data, saturation, xx, yy, layers)

    apply_border(layers)
    state.sky, state.sigma = fit_sky(data[~layers.border])

    state.saturated = detect_saturated(data, layers, saturation)
    state.bleeds = trace_bleeds(data, layers, state.saturated, state.sky, state.sigma)
    grow_bleed(data, layers, state.sky, state.sigma)

    state.stars = apply_halos(
        data, xx, yy, layers, state.saturated, saturation, state.sigma
    )
    return state


def build_mask(data: np.ndarray, header: Header) -> np.ndarray:
    return build_mask_state(data, header).mask