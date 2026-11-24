"""Command-line entry for photometric calibration (B2 lab)."""

from __future__ import annotations

import argparse
from pathlib import Path

from paths import (
    default_mosaic_path,
    default_sdss_stars_path,
    default_segmentation_path,
    pipeline_result_dir,
)
from pipeline.calibrate.pipeline import run_calibration


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Photometric calibration: SDSS zero point, galaxy injection, calibrated catalogue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "mosaic",
        nargs="?",
        type=Path,
        default=None,
        help=f"FITS mosaic (default: {default_mosaic_path()})",
    )
    p.add_argument(
        "catalogue",
        nargs="?",
        type=Path,
        default=None,
    )
    p.add_argument(
        "segmentation",
        nargs="?",
        type=Path,
        default=None,
    )
    p.add_argument(
        "mask",
        nargs="?",
        type=Path,
        default=None,
        help="mask .fits/.npy; if omitted, rebuild from mosaic",
    )
    p.add_argument(
        "photometry",
        nargs="?",
        type=Path,
        default=None,
    )
    p.add_argument(
        "sdss",
        nargs="?",
        type=Path,
        default=None,
        help=f"SDSS star CSV (default: {default_sdss_stars_path()})",
    )
    p.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=None,
        metavar="out_dir",
        help="Output directory (default: result/calibrate/<band>)",
    )
    p.add_argument("-o", "--outdir", type=Path, default=None, help="output directory")
    p.add_argument("--n-per-mag", type=int, default=300)
    p.add_argument("--quick", action="store_true", help="half the injection magnitude steps")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mosaic = args.mosaic or default_mosaic_path()
    detect_dir = pipeline_result_dir("detect", mosaic)
    args.catalogue = args.catalogue or (detect_dir / "catalogue.ecsv")
    args.segmentation = args.segmentation or default_segmentation_path(detect_dir)
    args.photometry = args.photometry or (
        pipeline_result_dir("photometry", mosaic) / "photometry.ecsv"
    )
    if args.outdir is not None:
        args.out_dir = args.outdir
    elif args.out_dir is None:
        args.out_dir = pipeline_result_dir("calibrate", mosaic)
    run_calibration(
        _require_file(mosaic, "mosaic"),
        _require_file(args.catalogue, "catalogue"),
        _require_file(args.segmentation, "segmentation"),
        _require_file(args.photometry, "photometry (run `python photometry.py` first)"),
        _require_file(args.sdss or default_sdss_stars_path(), "SDSS stars CSV (put SkyServer export in data/)"),
        Path(args.out_dir),
        mask_path=args.mask,
        n_per_mag=args.n_per_mag,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()