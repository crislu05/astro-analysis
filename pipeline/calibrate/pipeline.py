"""Orchestrate zero point, injection, and catalogue calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.table import Table
from photutils.segmentation import SegmentationImage, SourceCatalog

from pipeline.calibrate import config
from pipeline.calibrate.injection import (
    inject_galaxies,
    limit_at,
    measure_size_distribution,
    summarise_injection,
)
from pipeline.calibrate.log import log, reset_timer
from pipeline.calibrate.plots import save_calibration_plots
from pipeline.calibrate.zeropoint import zero_point
from pipeline.calibrate.zones import zone_of
from pipeline.detect import detect
from pipeline.detect.background import BackgroundResult, estimate_background
from pipeline.photometry.cog import build_empirical_psf, select_cog_stars
from pipeline.photometry.inputs import PhotometryInputs
from pipeline.photometry.io import load_detection_inputs, load_mask


def run_calibration(
    mosaic: Path,
    catalogue: Path,
    segmentation: Path,
    photometry_path: Path,
    sdss_path: Path,
    out_dir: Path,
    *,
    mask_path: Path | None = None,
    n_per_mag: int = 300,
    quick: bool = False,
) -> None:
    reset_timer()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_override = load_mask(mask_path) if mask_path is not None else None
    data, header, cat, segm_arr, mask, gain = load_detection_inputs(
        mosaic, catalogue, segmentation, mask=mask_override
    )
    phot = Table.read(photometry_path)
    if not np.all(np.asarray(cat["label"]) == np.asarray(phot["label"])):
        raise ValueError("catalogue and photometry rows differ")
    sdss = Table.read(sdss_path, format="ascii.csv", comment="#")
    ap_frac = float(phot.meta.get("AP_FRAC", 0.7334))

    bkg = estimate_background(data, mask)
    segm = SegmentationImage(segm_arr)
    kron_real = np.asarray(SourceCatalog(bkg.data_sub, segm, mask=mask).kron_flux, float)

    log("step 1: zero point from SDSS stars")
    zp, zp_plot = zero_point(data, header, mask, segm_arr, sdss, cat, phot, kron_real)

    log("step 2: galaxy injection")
    stars = select_cog_stars(cat, PhotometryInputs(data, mask, segm_arr, gain))
    psf, n_psf = build_empirical_psf(PhotometryInputs(data, mask, segm_arr, gain), stars)
    yy, xx = np.mgrid[
        -(psf.shape[0] // 2) : psf.shape[0] // 2 + 1,
        -(psf.shape[0] // 2) : psf.shape[0] // 2 + 1,
    ]
    rr = np.hypot(xx, yy)
    order = np.argsort(rr.ravel())
    cum = np.cumsum(psf.ravel()[order]) / psf.sum()
    psf_r50 = float(rr.ravel()[order][np.searchsorted(cum, 0.5)])
    with np.errstate(divide="ignore", invalid="ignore"):
        r_point = zp["ZP_ap"] - 2.5 * np.log10(np.asarray(phot["ap_flux"]))
    sizes, qs = measure_size_distribution(bkg.data_sub, segm, mask, r_point, psf_r50)
    log(
        f"PSF r50 = {psf_r50:.2f} px; {len(sizes)} resolved galaxies give intrinsic r_e: "
        f"median {np.median(sizes):.2f} px "
        f"(16-84%: {np.percentile(sizes, 16):.2f} to {np.percentile(sizes, 84):.2f})"
    )
    mags = config.INJ_MAGS[::2] if quick else config.INJ_MAGS
    inj = inject_galaxies(
        detect,
        BackgroundResult,
        data,
        mask,
        bkg,
        gain,
        psf,
        zp["ZP_ap"],
        zp["ZP_kron"],
        ap_frac,
        sizes,
        qs,
        mags,
        n_per_mag,
    )
    inj.write(out_dir / "injected_galaxies.ecsv", overwrite=True)
    comp, transfer = summarise_injection(inj)
    comp.write(out_dir / "completeness_galaxies.ecsv", overwrite=True)
    transfer.write(out_dir / "magnitude_transfer.ecsv", overwrite=True)

    lim90 = limit_at(comp, "all", 0.9)
    good = np.asarray(transfer["mag_in"]) <= lim90
    score = {}
    for est in ("m_ap", "m_kron"):
        b = np.asarray(transfer[f"{est}_bias"])[good]
        s = np.asarray(transfer[f"{est}_scatter"])[good]
        score[est] = float(np.sqrt(np.nanmean(b**2 + s**2)))
    best = min(score, key=score.get)
    log(
        f"estimator rms error over r <= {lim90:.2f}: point-source aperture {score['m_ap']:.3f}, "
        f"Kron {score['m_kron']:.3f} -> using {best}"
    )

    log("step 3: calibrated catalogue")
    with np.errstate(divide="ignore", invalid="ignore"):
        r_ap = zp["ZP_ap"] - 2.5 * np.log10(np.asarray(phot["ap_flux"]))
        r_kron = zp["ZP_kron"] - 2.5 * np.log10(kron_real)
    r_raw = r_ap if best == "m_ap" else r_kron
    bias_curve = np.asarray(transfer[f"{best}_bias"])
    m_curve = np.asarray(transfer["mag_in"])
    c_all = comp[comp["zone"] == "all"]
    c_at = np.interp(m_curve, np.asarray(c_all["mag"]), np.asarray(c_all["C"]))
    ok_curve = np.isfinite(bias_curve) & (c_at >= config.BIAS_MIN_COMPLETENESS)
    bias_here = np.interp(r_raw, m_curve[ok_curve], bias_curve[ok_curve])
    log(
        f"bias correction used for r <= {m_curve[ok_curve].max():.2f} "
        f"(C >= {config.BIAS_MIN_COMPLETENESS}); held constant at {bias_curve[ok_curve][-1]:+.3f} fainter"
    )
    r_gal = r_raw - bias_here
    out_cat = Table(
        {
            "label": phot["label"],
            "x": phot["x"],
            "y": phot["y"],
            "zone": zone_of(np.asarray(phot["x"]), np.asarray(phot["y"]), data.shape),
            "r_point": r_ap,
            "r_point_err": phot["mag_ap_err"],
            "r_kron_raw": r_kron,
            "r_gal": r_gal,
            "r_gal_bias_applied": bias_here,
            "beyond_90pc_complete": r_gal > lim90,
            "flag": phot["flag"],
        }
    )
    out_cat.meta.update(
        {
            "ZP_ap": zp["ZP_ap"],
            "ZP_kron": zp["ZP_kron"],
            "galaxy_estimator": best,
            "note": "r_point: point-source total magnitude (SDSS AB). r_gal: calibrated galaxy "
            "magnitude from the chosen estimator, bias-corrected with injected galaxies.",
        }
    )
    out_cat.write(out_dir / "calibrated_catalogue.ecsv", overwrite=True)

    summary = {
        "bias_correction_max_mag": float(m_curve[ok_curve].max()),
        "chance_match_fraction": float(len(cat) / (~mask).sum() * np.pi * config.MATCH_RADIUS**2),
        "zero_point": zp,
        "psf_r50_px": psf_r50,
        "n_psf_stars": n_psf,
        "injected_size_median_px": float(np.median(sizes)),
        "galaxy_estimator": best,
        "estimator_rms": score,
        "completeness_limits_galaxies": {
            z: {"r90": limit_at(comp, z, 0.9), "r50": limit_at(comp, z, 0.5)}
            for z in ("inner", "edge", "all")
        },
        "real_catalogue_median_kron_minus_point": {
            f"{lo}-{hi}": float(
                np.nanmedian((r_kron - r_ap)[(r_ap > lo) & (r_ap < hi) & (phot["flag"] == 0)])
            )
            for lo, hi in [(20, 22), (22, 23), (23, 24), (24, 25)]
        },
    }
    (out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=1, default=float) + "\n",
        encoding="utf-8",
    )
    log(json.dumps(summary["completeness_limits_galaxies"]))
    save_calibration_plots(out_dir, zp, zp_plot, comp, transfer, lim90, header)
    log("done")