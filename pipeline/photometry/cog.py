"""Curve of growth, stellar selection, and empirical PSF."""

import numpy as np
from astropy.table import Table
from scipy import ndimage

from pipeline.photometry import config
from pipeline.photometry.aperture import aperture_sum, local_sky
from pipeline.photometry.inputs import PhotometryInputs


def select_cog_stars(
    cat: Table, inp: PhotometryInputs, max_stars: int = 40, isolation: float = 30.0
):
    needed = {"snr", "ellipticity", "fwhm", "peak", "near_mask"}
    if not needed.issubset(cat.colnames):
        return Table()
    ok = (
        (cat["snr"] > 80)
        & (cat["ellipticity"] < 0.1)
        & (cat["peak"] < 20000)
        & ~np.asarray(cat["near_mask"], bool)
    )
    c = cat[ok]
    xs, ys = np.asarray(cat["x"]), np.asarray(cat["y"])
    keep = []
    for i, (x, y) in enumerate(zip(c["x"], c["y"])):
        d = np.hypot(xs - x, ys - y)
        if np.sum(d < isolation) == 1:
            keep.append(i)
    c = c[keep]
    c.sort("fwhm")
    return c[:max_stars]


def curve_of_growth(
    inp: PhotometryInputs,
    stars: Table,
    radii=config.COG_RADII,
    r_total=config.COG_R_TOTAL,
):
    curves, ap_std = [], []
    for x, y, lab in zip(stars["x"], stars["y"], stars["label"]):
        sky, sky_std, n_sky, bad = local_sky(
            inp, x, y, r_in=r_total + 5, r_out=r_total + 20
        )
        if bad or not np.isfinite(sky):
            continue
        fl = np.array([aperture_sum(inp, x, y, r, sky, lab)[0] for r in radii])
        f_tot = aperture_sum(inp, x, y, r_total, sky, lab)[0]
        sky_std_ann = local_sky(inp, x, y)[0]
        f_ap_std = aperture_sum(inp, x, y, config.R_AP, sky_std_ann, lab)[0]
        if f_tot > 0:
            curves.append(fl / f_tot)
            ap_std.append(f_ap_std / f_tot)
    return np.asarray(curves), np.asarray(ap_std)


def build_empirical_psf(
    inp: PhotometryInputs,
    stars: Table,
    half: int = 25,
    r_total: float = config.COG_R_TOTAL,
):
    stack = []
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    inside = np.hypot(xx, yy) <= r_total
    for x, y, lab in zip(stars["x"], stars["y"], stars["label"]):
        xi, yi = int(round(x)), int(round(y))
        cut = inp.data[yi - half : yi + half + 1, xi - half : xi + half + 1]
        if cut.shape != xx.shape:
            continue
        sky, _, _, bad = local_sky(
            inp, x, y, r_in=r_total + 5, r_out=r_total + 20
        )
        if bad:
            continue
        c = ndimage.shift(cut - sky, (yi - y, xi - x), order=3, mode="nearest")
        norm = c[inside].sum()
        if norm > 0:
            stack.append(c / norm)
    psf = np.median(np.asarray(stack), axis=0)
    return psf / psf[inside].sum(), len(stack)


def add_psf(image, x, y, flux, psf, gain=None, rng=None):
    half = psf.shape[0] // 2
    xi, yi = int(round(x)), int(round(y))
    stamp = flux * ndimage.shift(psf, (y - yi, x - xi), order=3, mode="constant")
    stamp = np.clip(stamp, 0, None)
    if gain is not None and rng is not None:
        stamp = rng.poisson(stamp * gain) / gain
    image[yi - half : yi + half + 1, xi - half : xi + half + 1] += stamp