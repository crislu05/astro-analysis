"""Local sky, aperture sums, and per-source photometry."""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clip
from astropy.table import Table
from photutils.aperture import CircularAnnulus, CircularAperture

from pipeline.photometry import config
from pipeline.photometry.inputs import PhotometryInputs


def _cutout(aperture_mask, image, fill):
    return aperture_mask.cutout(image, fill_value=fill)


def local_sky(
    inp: PhotometryInputs,
    x: float,
    y: float,
    r_in: float = config.R_IN,
    r_out: float = config.R_OUT,
):
    widened = False
    while True:
        ann = CircularAnnulus((x, y), r_in, r_out).to_mask(method="center")
        vals = _cutout(ann, inp.data, np.nan)
        bad = _cutout(ann, inp.not_sky.astype(float), 1.0) > 0
        use = (ann.data > 0) & ~bad & np.isfinite(vals)
        n = int(use.sum())
        if n >= config.MIN_SKY_PIX or r_out >= config.R_OUT_MAX:
            break
        r_out += config.R_OUT_STEP
        widened = True
    if n < 5:
        return np.nan, np.nan, n, True
    clipped = sigma_clip(
        vals[use], sigma=config.SKY_CLIP_SIGMA, maxiters=10, masked=False
    )
    return (
        float(np.median(clipped)),
        float(np.std(clipped)),
        int(clipped.size),
        widened or n < config.MIN_SKY_PIX,
    )


def aperture_sum(
    inp: PhotometryInputs,
    x: float,
    y: float,
    r: float,
    sky: float,
    own_label: int,
):
    ap = CircularAperture((x, y), r).to_mask(method="exact")
    w = ap.data
    vals = _cutout(ap, inp.data, np.nan)
    msk = _cutout(ap, inp.mask.astype(float), 1.0) > 0
    seg = _cutout(ap, inp.segm, 0)
    off_image = bool(np.any(~np.isfinite(vals) & (w > 0)))
    neighbour = (seg > 0) & (seg != own_label)
    vals = np.where(np.isfinite(vals), vals, sky)
    vals = np.where(msk | neighbour, sky, vals)
    total_w = w.sum()
    frac_mask = float((w * msk).sum() / total_w)
    frac_nb = float((w * (neighbour & ~msk)).sum() / total_w)
    flux = float((w * (vals - sky)).sum())
    area = float(total_w)
    return flux, area, frac_mask, frac_nb, off_image


def flux_error(flux, area, sky_std, n_sky, gain, corr_factor):
    poisson = max(flux, 0.0) / gain
    sky_var = corr_factor**2 * area * sky_std**2
    sky_est = corr_factor**2 * area**2 * sky_std**2 / max(n_sky, 1)
    return float(np.sqrt(poisson + sky_var + sky_est))


def run_photometry(
    inp: PhotometryInputs,
    x,
    y,
    labels,
    corr_factor: float = 1.0,
    r_ap: float = config.R_AP,
) -> Table:
    rows = []
    for xi, yi, lab in zip(
        np.asarray(x, float), np.asarray(y, float), np.asarray(labels, int)
    ):
        sky, sky_std, n_sky, sky_bad = local_sky(inp, xi, yi)
        flux, area, f_mask, f_nb, off = aperture_sum(inp, xi, yi, r_ap, sky, lab)
        err = (
            flux_error(flux, area, sky_std, n_sky, inp.gain, corr_factor)
            if np.isfinite(sky_std)
            else np.nan
        )
        flag = 0
        flag |= config.FLAG_MASKED if f_mask > 0 else 0
        flag |= config.FLAG_NEIGHBOUR if f_nb > config.NEIGHBOUR_FLAG_FRAC else 0
        flag |= config.FLAG_SKY if sky_bad else 0
        flag |= config.FLAG_NONPOSITIVE if not (flux > 0) else 0
        flag |= config.FLAG_EDGE if off else 0
        rows.append((lab, xi, yi, flux, err, sky, sky_std, n_sky, area, f_mask, f_nb, flag))
    names = [
        "label",
        "x",
        "y",
        "ap_flux",
        "ap_flux_err",
        "sky",
        "sky_std",
        "n_sky",
        "ap_area",
        "frac_masked",
        "frac_neighbour",
        "flag",
    ]
    return Table(rows=rows, names=names)