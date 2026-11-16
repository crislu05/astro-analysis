"""Per-star halo disks from radial surface-brightness profiles."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pipeline.mask import config
from pipeline.mask.geometry import mask_circle
from pipeline.mask.layers import MaskLayers
from pipeline.mask.saturated import SaturatedResult


@dataclass
class StarMask:
    xc: int
    yc: int
    mask_radius: float
    sat_pixels: int
    measured_radius: float
    peak: float | None = None


def halo_radius(
    data: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    layers: MaskLayers,
    xc: float,
    yc: float,
    sigma: float,
    r_max: int | None = None,
    dr: int | None = None,
    k: float | None = None,
    r_min: int | None = None,
) -> tuple[float, float]:
    r_max = config.HALO_R_MAX if r_max is None else r_max
    dr = config.HALO_DR if dr is None else dr
    k = config.HALO_K if k is None else k
    r_min = config.HALO_R_MIN if r_min is None else r_min
    ny, nx = data.shape
    x0, x1 = max(0, int(xc - r_max)), min(nx, int(xc + r_max) + 1)
    y0, y1 = max(0, int(yc - r_max)), min(ny, int(yc + r_max) + 1)
    cut = data[y0:y1, x0:x1]
    bad = (layers.saturated | layers.bleed)[y0:y1, x0:x1]
    r = np.hypot(xx[y0:y1, x0:x1] - xc, yy[y0:y1, x0:x1] - yc)
    radii, prof = [], []
    for r_in in np.arange(r_min, r_max, dr):
        ring = (r >= r_in) & (r < r_in + dr) & ~bad
        if ring.sum() < 5:
            continue
        radii.append(r_in + dr)
        prof.append(np.median(cut[ring]))
    if not radii:
        return float(r_max), float("nan")
    radii_arr, prof_arr = np.array(radii), np.array(prof)
    plateau = float(np.median(prof_arr[radii_arr >= 0.8 * r_max]))
    below = np.where(prof_arr < plateau + k * sigma)[0]
    r_meas = radii_arr[below[0]] if len(below) else float(r_max)
    return float(r_meas), plateau


def mask_saturated_halos(
    data: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    layers: MaskLayers,
    sat: SaturatedResult,
    sigma: float,
) -> list[StarMask]:
    stars: list[StarMask] = []
    safety = config.HALO_SAFETY
    for label_id, sl in enumerate(sat.slices, start=1):
        if sl is None:
            continue
        region = sat.labels[sl] == label_id
        row_widths = region.sum(axis=1)
        iy = int(np.argmax(row_widths))
        ix = int(np.median(np.where(region[iy])[0]))
        yc = sl[0].start + iy
        xc = sl[1].start + ix
        n_px = int(region.sum())
        r_max = (
            config.HALO_R_MAX_LARGE
            if n_px > config.HALO_LARGE_SAT_PX
            else config.HALO_R_MAX
        )
        r_meas, _ = halo_radius(data, xx, yy, layers, xc, yc, sigma, r_max=r_max)
        r = safety * r_meas
        mask_circle(layers.halo, xx, yy, xc, yc, r)
        stars.append(StarMask(xc, yc, r, n_px, r_meas))
    return stars


def mask_bright_unsaturated(
    data: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    layers: MaskLayers,
    saturation: float,
    sigma: float,
) -> list[StarMask]:
    cur = layers.combined()
    bright_peaks = (data > config.UNSAT_PEAK_FRAC * saturation) & ~cur
    pk_labels, _ = ndimage.label(bright_peaks)
    stars: list[StarMask] = []
    safety = config.HALO_SAFETY
    for i, sl in enumerate(ndimage.find_objects(pk_labels), start=1):
        sub = np.where(pk_labels[sl] == i, data[sl], -np.inf)
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        yc = sl[0].start + iy
        xc = sl[1].start + ix
        peak = float(data[yc, xc])
        r_meas, _ = halo_radius(
            data,
            xx,
            yy,
            layers,
            xc,
            yc,
            sigma,
            r_max=config.UNSAT_HALO_R_MAX,
        )
        r = safety * r_meas
        mask_circle(layers.halo, xx, yy, xc, yc, r)
        stars.append(StarMask(xc, yc, r, 0, r_meas, peak=peak))
    return stars


def apply_halos(
    data: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    layers: MaskLayers,
    sat: SaturatedResult,
    saturation: float,
    sigma: float,
) -> list[StarMask]:
    stars = mask_saturated_halos(data, xx, yy, layers, sat, sigma)
    stars.extend(
        mask_bright_unsaturated(data, xx, yy, layers, saturation, sigma)
    )
    return stars