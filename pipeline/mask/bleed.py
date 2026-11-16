"""Column bleed tracing and bright-region growth."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pipeline.mask import config
from pipeline.mask.layers import MaskLayers
from pipeline.mask.saturated import SaturatedResult


@dataclass
class BleedRecord:
    c_lo: int
    c_hi: int
    y_sat_start: int
    y_sat_stop: int
    y_bleed_bottom: int
    y_bleed_top: int


def trace_bleed(
    data: np.ndarray,
    sky: float,
    sigma: float,
    x_lo: int,
    x_hi: int,
    y_start: int,
    direction: int,
    ny: int,
    k: float | None = None,
    n_quiet: int | None = None,
) -> int:
    k = config.BLEED_SKY_K if k is None else k
    n_quiet = config.BLEED_QUIET_ROWS if n_quiet is None else n_quiet
    y = y_start
    last_bright = y_start
    quiet = 0
    while 0 <= y < ny:
        row_level = np.median(data[y, x_lo : x_hi + 1])
        if row_level < sky + k * sigma:
            quiet += 1
            if quiet >= n_quiet:
                break
        else:
            quiet = 0
            last_bright = y
        y += direction
    return last_bright


def trace_bleeds(
    data: np.ndarray,
    layers: MaskLayers,
    sat: SaturatedResult,
    sky: float,
    sigma: float,
) -> list[BleedRecord]:
    ny, nx = data.shape
    pad = config.BLEED_COL_PAD
    info: list[BleedRecord] = []
    for sl in sat.slices:
        if sl is None:
            continue
        y_sl, x_sl = sl
        x_lo = x_sl.start
        region = layers.saturated[sl]
        col_counts = region.sum(axis=0)
        core_cols = np.where(col_counts >= 0.5 * col_counts.max())[0] + x_lo
        c_lo, c_hi = int(core_cols.min()), int(core_cols.max())
        top = trace_bleed(data, sky, sigma, c_lo, c_hi, y_sl.stop - 1, +1, ny)
        bottom = trace_bleed(data, sky, sigma, c_lo, c_hi, y_sl.start, -1, ny)
        layers.bleed[
            bottom : top + 1,
            max(0, c_lo - pad) : min(nx, c_hi + pad + 1),
        ] = True
        info.append(
            BleedRecord(c_lo, c_hi, y_sl.start, y_sl.stop - 1, bottom, top)
        )
    return info


def grow_into_connected(
    data: np.ndarray,
    seed_mask: np.ndarray,
    sky: float,
    sigma: float,
    k: float | None = None,
) -> np.ndarray:
    k = config.GROW_BRIGHT_K if k is None else k
    bright = data > sky + k * sigma
    labels, _ = ndimage.label(bright)
    touching = np.unique(labels[seed_mask & bright])
    touching = touching[touching != 0]
    return np.isin(labels, touching)


def grow_bleed(
    data: np.ndarray,
    layers: MaskLayers,
    sky: float,
    sigma: float,
) -> None:
    seed = layers.saturated | layers.bleed
    layers.bleed |= grow_into_connected(data, seed, sky, sigma)