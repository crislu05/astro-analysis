"""Shape metrics and SDSS cross-matching."""

from __future__ import annotations

import numpy as np
from astropy.wcs import WCS

from pipeline.classifier import config


def measure_shapes(data_sub, segm_arr, mask, cat):
    from photutils.aperture import CircularAperture, aperture_photometry
    from photutils.segmentation import SegmentationImage, SourceCatalog

    sc = SourceCatalog(data_sub, SegmentationImage(segm_arr), mask=mask)
    labels = np.asarray(sc.label)
    r50_all = sc.fluxfrac_radius(0.5)
    r50_all = np.asarray(getattr(r50_all, "value", r50_all), float)
    lookup = dict(zip(labels, r50_all))
    r50 = np.array([lookup.get(int(label), np.nan) for label in cat["label"]])

    pos = np.c_[np.asarray(cat["x"]), np.asarray(cat["y"])]
    clean = np.where(mask, 0.0, data_sub)
    f_small = aperture_photometry(clean, CircularAperture(pos, config.R_SMALL))["aperture_sum"]
    f_big = aperture_photometry(clean, CircularAperture(pos, config.R_BIG))["aperture_sum"]
    with np.errstate(divide="ignore", invalid="ignore"):
        conc = -2.5 * np.log10(np.asarray(f_small) / np.asarray(f_big))
    return r50, conc


def centroid(data, x, y, half):
    from photutils.centroids import centroid_2dg

    a, b = int(round(x)), int(round(y))
    cut = data[b - half : b + half + 1, a - half : a + half + 1] - config.SKY_LEVEL
    if cut.shape != (2 * half + 1, 2 * half + 1) or cut.max() < 100:
        return np.nan, np.nan
    cx, cy = centroid_2dg(cut)
    return cx + a - half, cy + b - half


def match_reference(data, header, ref, cat):
    """Index into cat for each reference object (-1 if none)."""
    w = WCS(header)
    x0, y0 = w.all_world2pix(ref["ra"], ref["dec"], 0)
    cx, cy = np.asarray(cat["x"]), np.asarray(cat["y"])
    idx = np.full(len(ref), -1)
    for i, (a, b) in enumerate(zip(x0, y0)):
        xa, ya = centroid(data, a, b, 10)
        if not np.isfinite(xa):
            continue
        xa, ya = centroid(data, xa, ya, 6)
        if not np.isfinite(xa):
            continue
        d = np.hypot(cx - xa, cy - ya)
        j = int(np.argmin(d))
        if d[j] < config.MATCH_PX:
            idx[i] = j
    return idx