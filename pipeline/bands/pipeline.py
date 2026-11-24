"""Extract colour bands onto the lab mosaic grid for mask/detect/photometry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation

from paths import band_coverage_path, band_mosaic_path
from pipeline.bands.config import (
    DEFAULT_BANDS,
    DEFAULT_REFERENCE_BAND,
    DEFAULT_TILES,
    FILL_VALUE,
    MAX_OFFSET_RESIDUAL_PX,
    tile_path,
)
from pipeline.bands.headers import estimate_saturation, output_header
from pipeline.bands.io import load_lab_mosaic
from pipeline.bands.registration import fit_shift_model, measure_offsets, offset_check
from pipeline.bands.resample import build_band, identity_shift_model


@dataclass(frozen=True)
class ExtractConfig:
    lab_mosaic: Path
    tile_dir: Path
    tiles: tuple[str, ...]
    bands: tuple[str, ...]
    reference_band: str
    report_dir: Path


def extract_bands(config: ExtractConfig) -> dict:
    """Write mosaic_{band}.fits beside the lab mosaic; return extraction report."""
    config.report_dir.mkdir(parents=True, exist_ok=True)
    lab, lab_header, lab_wcs, shape = load_lab_mosaic(config.lab_mosaic)

    lab_saturation = float(lab_header.get("SATURATE", np.nanmax(lab)))
    lab_bad = ~np.isfinite(lab) | (lab <= 0) | (lab >= 0.9 * lab_saturation)
    lab_bad = binary_dilation(lab_bad, iterations=20)

    def tile_path_fn(tile: str, band: str) -> Path:
        return tile_path(config.tile_dir, tile, band)

    def archive_tiles_for(band: str) -> bool:
        return any(tile_path_fn(tile, band).is_file() for tile in config.tiles)

    report = {
        "lab_mosaic": str(config.lab_mosaic.resolve()),
        "tile_dir": str(config.tile_dir.resolve()),
        "reference_band": config.reference_band,
        "bands": {},
    }

    lab_is_r = config.reference_band.lower() == "r"
    fit_band = config.reference_band
    if lab_is_r and not archive_tiles_for("r"):
        print("1. WCS correction: lab mosaic_r.fits is the r band (no archive r tiles)")
        fit_band = next((b for b in config.bands if archive_tiles_for(b)), "")
        if fit_band:
            print(f"   Matching lab r to archive {fit_band} tiles to estimate offsets")
        else:
            print("   No archive tiles for any requested band; using zero shift correction")
            model = identity_shift_model(shape)
            report["shift_model"] = {
                "degree": 0,
                "held_out_median_px": None,
                "patches": 0,
                "coefficients": model.coefficients.tolist(),
                "source": "identity (no tiles)",
            }
            fit_band = None
    else:
        fit_band = config.reference_band
        print(f"1. Measuring the WCS correction with archive {fit_band} band")

    if fit_band:
        reference, _, _ = build_band(
            fit_band, config.tile_dir, config.tiles, tile_path_fn, lab_wcs, shape, None
        )
        positions, shifts = measure_offsets(lab, reference, lab_bad)
        if len(positions):
            raw_median = float(np.median(np.linalg.norm(shifts, axis=1)))
            print(f"  {len(positions)} patches matched, raw median offset {raw_median:.2f} px")
        else:
            print("  no patches matched; using zero shift correction")
        model, degree, cv_residual = fit_shift_model(positions, shifts, shape)
        report["shift_model"] = {
            "degree": degree,
            "held_out_median_px": cv_residual,
            "patches": len(positions),
            "coefficients": model.coefficients.tolist(),
            "source": f"lab r vs archive {fit_band}",
        }

    if lab_is_r and not archive_tiles_for("r"):
        print("2. Reference band r: using lab mosaic (not rewritten)")
        report["reference_r"] = {"file": str(config.lab_mosaic.resolve()), "source": "lab"}
    elif archive_tiles_for(config.reference_band):
        print("2. Checking the correction on the reference band")
        reference, _, _ = build_band(
            config.reference_band,
            config.tile_dir,
            config.tiles,
            tile_path_fn,
            lab_wcs,
            shape,
            model,
        )
        residual, n = offset_check(lab, reference, lab_bad)
        print(f"  {config.reference_band}: residual offset {residual:.2f} px from {n} patches")
        if not residual < MAX_OFFSET_RESIDUAL_PX:
            raise RuntimeError("WCS correction failed on the reference band; inspect before continuing.")
    else:
        print(f"2. No archive tiles for band {config.reference_band}; skip reference check")

    print("3. Extracting bands")
    for band in config.bands:
        if band.lower() == "r":
            print("  r: lab mosaic_r.fits is the r band, skipped")
            report["bands"]["r"] = {
                "written": False,
                "file": str(config.lab_mosaic.resolve()),
                "source": "lab mosaic",
            }
            continue
        if not archive_tiles_for(band):
            print(f"  {band}: no tiles found, skipped")
            continue
        image, band_header, tiles_used = build_band(
            band, config.tile_dir, config.tiles, tile_path_fn, lab_wcs, shape, model
        )
        coverage = np.isfinite(image)
        lab_area = ~lab_bad
        covered = float((coverage & lab_area).sum() / lab_area.sum())
        residual, n = offset_check(lab, image, lab_bad)
        saturation, plateau = estimate_saturation(image)
        print(
            f"  {band}: coverage {covered:.1%}, residual offset {residual:.2f} px ({n} patches), "
            f"max {saturation:.1f} ({plateau} px within 1%)"
        )

        if not residual < MAX_OFFSET_RESIDUAL_PX:
            print(f"  {band}: residual offset too large, NOT written")
            report["bands"][band] = {"written": False, "residual_offset_px": residual}
            continue

        sky = np.median(image[coverage])
        noise = 1.4826 * np.median(np.abs(image[coverage] - sky))
        out = band_mosaic_path(band, config.lab_mosaic)
        header = output_header(
            lab_header, band_header, band, (saturation, plateau), tiles_used, config.lab_mosaic
        )
        data = np.where(coverage, image, FILL_VALUE).astype(np.float32)
        fits.PrimaryHDU(data, header).writeto(out, overwrite=True)
        coverage_path = band_coverage_path(band)
        fits.PrimaryHDU(coverage.astype(np.uint8)).writeto(coverage_path, overwrite=True)
        print(
            f"  {band}: wrote {out}  (sky ~ {sky:.2f}, pixel rms ~ {noise:.2f}, "
            f"MAGZPT {header.get('MAGZPT')}, EXPTIME {header.get('EXPTIME')})"
        )
        report["bands"][band] = {
            "written": True,
            "file": str(out),
            "coverage": covered,
            "residual_offset_px": residual,
            "sky_median": float(sky),
            "pixel_rms": float(noise),
            "MAGZPT": header.get("MAGZPT"),
            "EXPTIME": header.get("EXPTIME"),
            "GAIN": header.get("GAIN"),
            "SATURATE": header.get("SATURATE"),
            "tiles": tiles_used,
        }

    report_path = config.report_dir / "extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Done. Report: {report_path}")
    return report


def default_extract_config(
    lab_mosaic: Path,
    tile_dir: Path,
    report_dir: Path,
    bands: tuple[str, ...] | None = None,
) -> ExtractConfig:
    return ExtractConfig(
        lab_mosaic=lab_mosaic,
        tile_dir=tile_dir,
        tiles=DEFAULT_TILES,
        bands=bands or DEFAULT_BANDS,
        reference_band=DEFAULT_REFERENCE_BAND,
        report_dir=report_dir,
    )