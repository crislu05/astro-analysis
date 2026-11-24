"""Output headers aligned with the lab mosaic for downstream pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.bands.config import CALIBRATION_KEYS, FILL_VALUE


def estimate_saturation(image):
    good = image[np.isfinite(image)]
    top = good.max()
    plateau = np.sum(good >= 0.99 * top)
    return float(top), int(plateau)


def output_header(lab_header, band_header, band: str, saturation, tiles_used, lab_path: Path):
    """Lab WCS layout with per-band calibration from the archive."""
    header = lab_header.copy()
    for key in ["BZERO", "BSCALE", "BLANK"]:
        header.remove(key, ignore_missing=True)

    for key in CALIBRATION_KEYS:
        if key in band_header:
            header[key] = band_header[key]
        elif key != "SATURATE" and key in header:
            print(f"  WARNING: {key} not in archive header; lab value {header[key]} kept, CHECK IT")

    if "SATURATE" not in band_header:
        value, plateau = saturation
        header["SATURATE"] = (value, "estimated from image maximum")
        if plateau < 20:
            print(f"  WARNING: no saturation plateau found; SATURATE = image max {value:.1f}, CHECK IT")
    header["FILTER"] = band_header.get("FILTER", band)
    header["BAND"] = band.lower()
    header["REFFRAME"] = (lab_path.name, "pixel grid and WCS copied from")
    for index, tile in enumerate(tiles_used, start=1):
        header[f"SRCFILE{index}"] = tile["file"]
        header[f"ZPSCALE{index}"] = (tile["zp_scale"], "flux scale applied to this tile")
    header["MISSVAL"] = (FILL_VALUE, "value of pixels with no archive data")
    header["HISTORY"] = "Bilinear resampling x pixel-area Jacobian onto the lab grid,"
    header["HISTORY"] = "with an empirical polynomial WCS correction measured on r band."
    return header