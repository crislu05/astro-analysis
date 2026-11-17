"""Run aperture photometry on a detection catalogue."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.table import Table

from pipeline.photometry import config
from pipeline.photometry.aperture import run_photometry
from pipeline.photometry.calibration import counts_to_mag, mag_error
from pipeline.photometry.cog import (
    build_empirical_psf,
    curve_of_growth,
    select_cog_stars,
)
from pipeline.photometry.correlation import empty_aperture_factor
from pipeline.photometry.injection import validate_injection
from pipeline.photometry.inputs import PhotometryInputs
from pipeline.photometry.plots import save_photometry_plots


def run_photometry_catalogue(
    inp: PhotometryInputs,
    cat: Table,
    *,
    zp: float,
    zp_err: float,
    exptime: float,
    out: Path,
    validate: bool = False,
) -> tuple[Table, dict]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    corr, measured, expected, empty_sums = empty_aperture_factor(inp)
    print(
        f"empty apertures: scatter {measured:.1f} counts vs {expected:.1f} expected "
        f"for independent pixels -> correlated-noise factor {corr:.2f}"
    )

    phot = run_photometry(inp, cat["x"], cat["y"], cat["label"], corr_factor=corr)

    stars = select_cog_stars(cat, inp)
    cog, frac = (
        curve_of_growth(inp, stars) if len(stars) else (np.empty((0, len(config.COG_RADII))), np.empty(0))
    )
    if len(cog):
        ap_frac = float(np.median(frac))
        ap_frac_err = float(1.2533 * np.std(frac) / np.sqrt(len(frac)))
        i_ap = int(np.argmin(abs(config.COG_RADII - config.R_AP)))
        print(
            f"  light inside r = {config.R_AP} px with a far sky: {100 * np.median(cog[:, i_ap]):.1f}%; "
            f"as measured by the pipeline (standard annulus): {100 * ap_frac:.1f}%"
        )
    else:
        ap_frac, ap_frac_err = np.nan, np.nan
    ap_corr_mag = -2.5 * np.log10(ap_frac) if np.isfinite(ap_frac) else 0.0
    print(
        f"curve of growth from {len(cog)} stars: {100 * ap_frac:.1f}% of a star's light "
        f"inside r = {config.R_AP} px -> aperture correction {ap_corr_mag:.3f} mag"
    )

    phot["mag_ap"] = counts_to_mag(phot["ap_flux"], zp)
    phot["mag_ap_err"] = mag_error(phot["ap_flux"], phot["ap_flux_err"])
    phot["mag_psf_total"] = phot["mag_ap"] - ap_corr_mag
    phot["mag_rate_ap"] = counts_to_mag(phot["ap_flux"], zp, exptime)
    phot["snr_ap"] = phot["ap_flux"] / phot["ap_flux_err"]
    for col in ("near_mask", "snr", "fwhm", "ellipticity", "flux"):
        if col in cat.colnames:
            phot[f"det_{col}"] = cat[col]
    phot.meta.update(
        {
            "R_AP": config.R_AP,
            "R_IN": config.R_IN,
            "R_OUT": config.R_OUT,
            "MAGZPT": zp,
            "MAGZRR": zp_err,
            "EXPTIME": exptime,
            "GAIN": inp.gain,
            "CORR_FACTOR": corr,
            "AP_FRAC": ap_frac,
            "AP_CORR_MAG": ap_corr_mag,
            "NOTE": "mag_ap uses lab eq. 5; mag_rate_ap uses counts/EXPTIME; MAGZRR is systematic",
        }
    )
    phot.write(out / "photometry.ecsv", overwrite=True)

    clean = phot["flag"] == 0
    summary = {
        "n_sources": len(phot),
        "n_clean_flag0": int(clean.sum()),
        "flag_counts": {
            name: int(np.sum(phot["flag"] & bit > 0))
            for name, bit in [
                ("masked", config.FLAG_MASKED),
                ("neighbour", config.FLAG_NEIGHBOUR),
                ("sky", config.FLAG_SKY),
                ("nonpositive", config.FLAG_NONPOSITIVE),
                ("edge", config.FLAG_EDGE),
            ]
        },
        "correlated_noise_factor": round(corr, 3),
        "empty_aperture_scatter_counts": round(measured, 2),
        "aperture_fraction": round(ap_frac, 4),
        "aperture_fraction_err": round(ap_frac_err, 4),
        "aperture_correction_mag": round(ap_corr_mag, 4),
        "n_cog_stars": int(len(cog)),
        "median_sky": round(float(np.nanmedian(phot["sky"])), 2),
        "mag_ap_range_p1_p99": [
            round(float(v), 2) for v in np.nanpercentile(phot["mag_ap"], [1, 99])
        ],
        "sn5_limit_mag_ap": round(
            float(
                np.nanmedian(
                    phot["mag_ap"][(phot["snr_ap"] > 4.5) & (phot["snr_ap"] < 5.5)]
                )
            ),
            2,
        ),
        "zero_point": zp,
        "zero_point_err": zp_err,
    }
    print(json.dumps(summary, indent=1))

    if validate and len(stars):
        psf, n_psf = build_empirical_psf(inp, stars)
        summary["n_psf_stars"] = n_psf
        summary["injection"] = validate_injection(
            inp, cat, zp, ap_corr_mag, corr, out, psf=psf
        )

    (out / "photometry_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    save_photometry_plots(out, phot, cog, ap_frac, empty_sums, expected, measured, summary)
    return phot, summary