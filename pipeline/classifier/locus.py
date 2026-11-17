"""Stellar locus fit and width in magnitude bins."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from pipeline.classifier import config
from pipeline.classifier.log import log


def robust_sigma(values) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def stellar_locus(mag, metric, star_idx, flag):
    s = star_idx[star_idx >= 0]
    s = s[(flag[s] == 0) & np.isfinite(metric[s]) & np.isfinite(mag[s])]
    x, y = mag[s], metric[s]
    keep = np.ones(len(x), bool)
    for _ in range(5):
        p = np.polyfit(x[keep], y[keep], 1)
        res = y - np.polyval(p, x)
        sig_sdss = robust_sigma(res[keep])
        keep = np.abs(res) < 3 * sig_sdss

    def locus(m):
        return np.polyval(p, m)

    edges = np.arange(config.MAG_RANGE[0], config.MAG_RANGE[1] + config.BIN, config.BIN)
    rows = []
    for lo in edges[:-1]:
        hi = lo + config.BIN
        inb = (mag >= lo) & (mag < hi) & (flag == 0) & np.isfinite(metric)
        d = metric[inb] - locus(mag[inb])
        below = d[d < 0]
        sig = float(np.sqrt(np.mean(below**2))) if len(below) >= 5 else np.nan
        gal = d[d > 0]
        rows.append(
            {
                "mag_lo": lo,
                "mag_hi": hi,
                "n": int(inb.sum()),
                "sigma": sig,
                "galaxy_median_offset": float(np.median(gal)) if len(gal) >= 5 else np.nan,
            }
        )
    tab = Table(rows=rows)
    sig = np.asarray(tab["sigma"], float)
    sig = np.fmax.accumulate(np.nan_to_num(np.fmax(sig, sig_sdss), nan=sig_sdss))
    tab["sigma_used"] = sig
    tab["separation"] = np.asarray(tab["galaxy_median_offset"]) / sig
    tab.meta["locus_coeffs"] = [float(v) for v in p]
    log(
        f"stellar locus: c = {p[1]:.3f} + {p[0]:.4f} (r - 0) from {keep.sum()} SDSS stars, "
        f"scatter {sig_sdss:.3f}"
    )
    return locus, sig_sdss, tab