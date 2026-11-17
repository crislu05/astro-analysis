"""SDSS reference star counts and extrapolated model."""

from __future__ import annotations

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

from pipeline.classifier import config


def sdss_counts_unmasked(header, mask, ref):
    x, y = WCS(header).all_world2pix(ref["ra"], ref["dec"], 0)
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    inside = (xi >= 0) & (xi < mask.shape[1]) & (yi >= 0) & (yi < mask.shape[0])
    good = np.zeros(len(ref), bool)
    good[inside] = ~mask[yi[inside], xi[inside]]
    edges = np.arange(config.MAG_RANGE[0], config.MAG_RANGE[1] + config.BIN, config.BIN)
    return np.histogram(np.asarray(ref["psfMag_r"])[good], edges)[0]


def star_count_model(mag, cls, area_deg2, mag_limit, n_ref):
    edges = np.arange(config.MAG_RANGE[0], config.MAG_RANGE[1] + config.BIN, config.BIN)
    centres = edges[:-1] + config.BIN / 2
    n_class = np.histogram(mag[cls == "star"], edges)[0]
    fit = (
        (centres > config.STAR_FIT_RANGE_MIN)
        & (edges[1:] <= config.SDSS_STAR_LIMIT)
        & (n_ref > 0)
    )
    if fit.sum() >= 2:
        w = np.sqrt(n_ref[fit]) / 0.434
        p, cov = np.polyfit(centres[fit], np.log10(n_ref[fit]), 1, w=w, cov="unscaled")
        slope, slope_err = float(p[0]), float(np.sqrt(cov[0, 0]))
        pred = 10 ** np.polyval(p, centres)
    else:
        slope, slope_err, pred = np.nan, np.nan, np.full(len(centres), np.nan)
    excess = np.where(edges[1:] <= config.SDSS_STAR_LIMIT, n_class - n_ref, -1)
    tab = Table(
        {
            "mag_lo": edges[:-1],
            "mag_hi": edges[1:],
            "n_star_sdss": n_ref,
            "n_star_classified": n_class,
            "classified_minus_sdss": excess,
            "n_star_model": pred,
            "per_deg2_per_0p5mag": pred / area_deg2,
            "in_fit": fit,
            "classified_range": centres < mag_limit,
        }
    )
    return tab, slope, slope_err