"""Command-line entry for photometric calibration (B2 lab)."""

from __future__ import annotations

import argparse
from pathlib import Path

from paths import (
    CALIBRATE_RESULT_DIR,
    DETECT_RESULT_DIR,
    PHOTOMETRY_RESULT_DIR,
    default_mosaic_path,
    default_sdss_stars_path,
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
        default=DETECT_RESULT_DIR / "catalogue.ecsv",
    )
    p.add_argument(
        "segmentation",
        nargs="?",
        type=Path,
        default=DETECT_RESULT_DIR / "segmentation.fits",
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
        default=PHOTOMETRY_RESULT_DIR / "photometry.ecsv",
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
        help=f"Output directory (default: {CALIBRATE_RESULT_DIR})",
    )
    p.add_argument("-o", "--outdir", type=Path, default=None, help="output directory")
    p.add_argument("--n-per-mag", type=int, default=300)
    p.add_argument("--quick", action="store_true", help="half the injection magnitude steps")
    args = p.parse_args(argv)
    if args.outdir is not None:
        args.out_dir = args.outdir
    elif args.out_dir is None:
        args.out_dir = CALIBRATE_RESULT_DIR
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_calibration(
        _require_file(args.mosaic or default_mosaic_path(), "mosaic"),
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