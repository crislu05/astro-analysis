"""Injected-star validation for photometry bias and errors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy import ndimage

from pipeline.photometry import config
from pipeline.photometry.aperture import aperture_sum, local_sky, run_photometry
from pipeline.photometry.calibration import counts_to_mag, mag_error
from pipeline.photometry.cog import add_psf
from pipeline.photometry.inputs import PhotometryInputs


def validate_injection(
    inp: PhotometryInputs,
    cat: Table,
    zp: float,
    ap_corr_mag: float,
    corr: float,
    out: Path,
    psf: np.ndarray,
    mags=np.arange(13.0, 19.01, 0.5),
    n_per_mag: int = 150,
):
    rng = np.random.default_rng(7)
    ny, nx = inp.data.shape
    half = psf.shape[0] // 2
    clear = ~ndimage.binary_dilation(inp.not_sky, iterations=12)
    size = 2 * (half + int(config.R_OUT_MAX)) + 1
    blank = np.zeros((size, size))
    c0 = size // 2
    add_psf(blank, c0, c0, 1.0, psf)
    seg_b = np.zeros((size, size), int)
    blank_in = PhotometryInputs(blank, np.zeros_like(blank, bool), seg_b, inp.gain)
    sky_b = local_sky(blank_in, c0, c0)[0]
    psf_ap_frac = aperture_sum(blank_in, c0, c0, config.R_AP, sky_b, own_label=1)[0]
    inj_corr_mag = -2.5 * np.log10(psf_ap_frac)
    print(
        f"  injected PSF: {100 * psf_ap_frac:.1f}% measured in the aperture "
        f"(real stars: {100 * 10 ** (-ap_corr_mag / 2.5):.1f}%), "
        f"PSF-model offset {inj_corr_mag - ap_corr_mag:+.3f} mag"
    )
    rows = []
    for m in mags:
        F = 10 ** ((zp - m) / 2.5)
        pts = []
        while len(pts) < n_per_mag:
            x, y = rng.uniform(60, nx - 60), rng.uniform(60, ny - 60)
            if clear[int(y), int(x)] and all(np.hypot(x - a, y - b) > 70 for a, b in pts):
                pts.append((x, y))
        pts = np.array(pts)
        img = inp.data.copy()
        seg = inp.segm.copy()
        label0 = seg.max() + 1
        labels = []
        for k, (x, y) in enumerate(pts):
            add_psf(img, x, y, F, psf, gain=inp.gain, rng=rng)
            yy, xx = np.mgrid[int(y) - 10 : int(y) + 11, int(x) - 10 : int(x) + 11]
            seg[yy, xx] = np.where(np.hypot(xx - x, yy - y) < 7, label0 + k, seg[yy, xx])
            labels.append(label0 + k)
        test = PhotometryInputs(img, inp.mask, seg, inp.gain)
        t = run_photometry(test, pts[:, 0], pts[:, 1], labels, corr_factor=corr)
        m_out = counts_to_mag(t["ap_flux"], zp) - inj_corr_mag
        pull = (t["ap_flux"] - F * psf_ap_frac) / t["ap_flux_err"]
        ok = np.isfinite(m_out)
        rows.append(
            {
                "mag_in": float(m),
                "n": int(len(pts)),
                "median_dmag": float(np.median(m_out[ok] - m)),
                "dmag_scatter": float(
                    1.4826 * np.median(np.abs(m_out[ok] - np.median(m_out[ok])))
                ),
                "median_mag_err": float(
                    np.median(mag_error(t["ap_flux"], t["ap_flux_err"])[ok])
                ),
                "pull_mean": float(np.mean(pull)),
                "pull_std": float(np.std(pull)),
                "pull_std_robust": float(
                    1.4826 * np.median(np.abs(pull - np.median(pull)))
                ),
                "frac_nonpositive": float(np.mean(~ok)),
            }
        )
        r = rows[-1]
        print(
            f"  inject m={m:5.2f}: dm={r['median_dmag']:+.3f}  scatter {r['dmag_scatter']:.3f}  "
            f"quoted err {r['median_mag_err']:.3f}  pull {r['pull_mean']:+.2f}, "
            f"std {r['pull_std']:.2f} (robust {r['pull_std_robust']:.2f})"
        )
    (out / "injection_photometry.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows