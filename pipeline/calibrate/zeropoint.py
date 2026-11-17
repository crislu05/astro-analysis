"""SDSS-based aperture and Kron zero points."""

from __future__ import annotations

import numpy as np
from astropy.wcs import WCS

from pipeline.calibrate import config
from pipeline.calibrate.log import log
from pipeline.calibrate.stats import robust
from pipeline.photometry import config as pconfig
from pipeline.photometry.aperture import run_photometry
from pipeline.photometry.inputs import PhotometryInputs


def centroid_star(data, x, y, half):
    from photutils.centroids import centroid_2dg

    a, b = int(round(x)), int(round(y))
    cut = data[b - half : b + half + 1, a - half : a + half + 1] - config.SKY_LEVEL
    if cut.shape != (2 * half + 1, 2 * half + 1) or cut.max() < 150:
        return np.nan, np.nan, np.nan
    cx, cy = centroid_2dg(cut)
    return cx + a - half, cy + b - half, cut.max() + config.SKY_LEVEL


def zero_point(data, header, mask_hard, segm, sdss, cat, phot, kron_flux):
    """Aperture and Kron zero points on the SDSS (AB) scale."""
    wcs = WCS(header)
    ny, nx = data.shape
    sat = float(header["SATURATE"])
    x0, y0 = wcs.all_world2pix(sdss["ra"], sdss["dec"], 0)
    first = np.array([centroid_star(data, a, b, 10) for a, b in zip(x0, y0)])
    final = np.array(
        [
            centroid_star(data, a, b, 6) if np.isfinite(a) else (np.nan,) * 3
            for a, b in zip(first[:, 0], first[:, 1])
        ]
    )
    x, y, peak = final.T
    ok = np.isfinite(x) & (x > 160) & (x < nx - 160) & (y > 160) & (y < ny - 160)
    ok &= np.hypot(x - x0, y - y0) < 10
    xi = np.where(ok, np.round(x), 0).astype(int)
    yi = np.where(ok, np.round(y), 0).astype(int)
    ok &= ~mask_hard[yi, xi]
    gain = float(header.get("GAIN", 3.1))
    inp = PhotometryInputs(data, mask_hard, segm, gain)
    t = run_photometry(inp, x[ok], y[ok], segm[yi[ok], xi[ok]], corr_factor=config.AP_CORR_FACTOR)
    r = np.asarray(sdss["psfMag_r"])[ok]
    clean = ((np.asarray(t["flag"]) & ~pconfig.FLAG_MASKED) == 0) & (np.asarray(t["ap_flux"]) > 0)
    clean &= peak[ok] < config.SAT_FRAC * sat
    use = clean & (r > config.ZP_STAR_RANGE[0]) & (r < config.ZP_STAR_RANGE[1])
    zp_vals = r[use] + 2.5 * np.log10(np.asarray(t["ap_flux"])[use])
    zp_ap, zp_ap_sig, zp_ap_err = robust(zp_vals)
    trend = np.polyfit(r[use], zp_vals, 1)[0]
    log(
        f"ZP_ap = {zp_ap:.3f} +- {zp_ap_err:.3f} (n = {use.sum()}, scatter {zp_ap_sig:.3f}, "
        f"trend {trend:+.4f} mag/mag)"
    )

    cx, cy = np.asarray(cat["x"]), np.asarray(cat["y"])
    kron_vals, kron_r = [], []
    for xs, ys, rs, pk in zip(x[ok], y[ok], r, peak[ok]):
        d = np.hypot(cx - xs, cy - ys)
        j = int(np.argmin(d))
        if (
            d[j] < 2.0
            and phot["flag"][j] == 0
            and kron_flux[j] > 0
            and pk < config.SAT_FRAC * sat
            and config.KRON_STAR_RANGE[0] < rs < config.KRON_STAR_RANGE[1]
        ):
            kron_vals.append(rs + 2.5 * np.log10(kron_flux[j]))
            kron_r.append(rs)
    zp_kron, zp_kron_sig, zp_kron_err = robust(kron_vals)
    log(f"ZP_kron = {zp_kron:.3f} +- {zp_kron_err:.3f} (n = {len(kron_vals)}, scatter {zp_kron_sig:.3f})")
    result = {
        "ZP_ap": zp_ap,
        "ZP_ap_err": zp_ap_err,
        "ZP_ap_scatter": zp_ap_sig,
        "n_ap": int(use.sum()),
        "ZP_ap_trend_mag_per_mag": float(trend),
        "ZP_kron": zp_kron,
        "ZP_kron_err": zp_kron_err,
        "ZP_kron_scatter": zp_kron_sig,
        "n_kron": len(kron_vals),
        "offset_vs_lab_formula": zp_ap - float(header["MAGZPT"]),
    }
    plot = {
        "r": r[use],
        "zp": zp_vals,
        "kron_r": np.array(kron_r),
        "kron_zp": np.array(kron_vals),
    }
    return result, plot