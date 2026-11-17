"""Build mask, detect sources, validate, and write results."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from astropy.io import fits

from paths import DETECT_RESULT_DIR, default_mosaic_path
from pipeline.detect import config as dconfig
from pipeline.detect import detect
from pipeline.detect.plots import save_validation_plots
from pipeline.detect.psf import measure_stellar_fwhm
from pipeline.detect.util import instrumental_mag
from pipeline.detect.validation import (
    detection_summary,
    injection_recovery,
    run_negative_image_test,
    spatial_uniformity,
)
from pipeline.mask import build_mask_state

_t0 = time.time()


def log(*args) -> None:
    print(f"[{time.time() - _t0:6.1f}s]", *args, flush=True)


def run_detect_validation(mosaic: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with fits.open(mosaic) as hdul:
        hdu = hdul[0]
        data = hdu.data.astype(float)
        header = hdu.header
    gain = float(header.get("GAIN", dconfig.DEFAULT_GAIN))
    zp = float(header["MAGZPT"])
    pix_arcsec = float(header.get("PIXSCAL1", 0.258))
    results: dict = {"zero_point": zp, "gain": gain}

    state = build_mask_state(data, header)
    mask = state.mask
    good_px = int((~mask).sum())
    area_deg2 = good_px * (pix_arcsec / 3600) ** 2
    results["mask"] = {
        "masked_percent": round(100 * mask.mean(), 2),
        "good_pixels": good_px,
        "good_area_deg2": round(area_deg2, 5),
    }
    log(f"mask built: {100 * mask.mean():.1f}% masked, good area {area_deg2:.4f} deg^2")

    res = detect(data, mask, gain=gain)
    cat = res.catalogue
    cat["mag"] = instrumental_mag(cat["flux"], zp)
    cat.write(outdir / "catalogue.ecsv", overwrite=True)
    seg_data = (
        res.segmentation.segm.data.astype(np.int32)
        if res.segmentation.segm is not None
        else np.zeros(data.shape, dtype=np.int32)
    )
    fits.PrimaryHDU(seg_data).writeto(outdir / "segmentation.fits", overwrite=True)
    n_real = len(cat)
    results["detection"] = detection_summary(res, area_deg2, good_px)
    log(f"detected {n_real} sources")

    stellar_fwhm, fwhm_fits = measure_stellar_fwhm(data, cat)
    results["psf"] = {
        "n_stars_fitted": len(fwhm_fits),
        "stellar_fwhm_px": round(stellar_fwhm, 2),
        "stellar_fwhm_arcsec": round(stellar_fwhm * pix_arcsec, 2),
        "fwhm_p10_p90_px": [round(float(v), 2) for v in np.percentile(fwhm_fits, [10, 90])]
        if len(fwhm_fits)
        else [],
    }
    log(f"stellar FWHM ~ {stellar_fwhm:.2f} px ({stellar_fwhm * pix_arcsec:.2f} arcsec)")

    neg_block, neg_cat = run_negative_image_test(data, mask, res.background, gain, zp, n_real, good_px)
    results["negative_test"] = neg_block
    log(
        f"negative image: {neg_block['n_false']} detections "
        f"({100 * neg_block['n_false'] / n_real:.2f}% of real)"
    )

    rec_rows, limits = injection_recovery(
        data, mask, cat, res.background, gain, zp, stellar_fwhm, log
    )
    results["injection"] = rec_rows
    results["completeness_limits"] = limits
    log(f"90% complete to m={limits['mag_90']:.2f}, 50% to m={limits['mag_50']:.2f}")

    mag_90 = limits["mag_90"] if np.isfinite(limits["mag_90"]) else 99.0
    results["uniformity"] = spatial_uniformity(data, mask, cat, res.background, mag_90)
    log("uniformity:", results["uniformity"])

    (outdir / "validation.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )
    save_validation_plots(
        outdir, data, mask, res, cat, neg_cat, rec_rows, results, zp
    )
    log("done")


def main(argv: list[str] | None = None) -> None:
    global _t0
    _t0 = time.time()
    parser = argparse.ArgumentParser(
        description="Mask, detect sources on the mosaic, and run validation checks.",
    )
    parser.add_argument(
        "mosaic",
        nargs="?",
        type=Path,
        default=None,
        help=f"Input FITS (default: {default_mosaic_path()})",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=DETECT_RESULT_DIR,
        help=f"Output directory (default: {DETECT_RESULT_DIR})",
    )
    args = parser.parse_args(argv)
    args.mosaic = args.mosaic or default_mosaic_path()
    if not args.mosaic.is_file():
        raise SystemExit(f"mosaic not found: {args.mosaic}")
    run_detect_validation(args.mosaic, args.outdir)


if __name__ == "__main__":
    main()