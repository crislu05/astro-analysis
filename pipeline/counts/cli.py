"""Command-line entry for galaxy number counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from paths import (
    CALIBRATE_RESULT_DIR,
    CLASSIFIER_RESULT_DIR,
    NUMBER_COUNTS_RESULT_DIR,
    default_mosaic_path,
)
from pipeline.counts.pipeline import run_number_counts


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Differential galaxy number counts with completeness and star subtraction.",
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
        default=CLASSIFIER_RESULT_DIR / "classified_catalogue.ecsv",
    )
    p.add_argument(
        "completeness",
        nargs="?",
        type=Path,
        default=CALIBRATE_RESULT_DIR / "completeness_galaxies.ecsv",
    )
    p.add_argument(
        "star_model",
        nargs="?",
        type=Path,
        default=CLASSIFIER_RESULT_DIR / "star_count_model.ecsv",
    )
    p.add_argument(
        "mask",
        nargs="?",
        type=Path,
        default=None,
        help="mask .fits/.npy; if omitted, rebuild from mosaic",
    )
    p.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=None,
        metavar="out_dir",
        help=f"Output directory (default: {NUMBER_COUNTS_RESULT_DIR})",
    )
    p.add_argument("-o", "--outdir", type=Path, default=None)
    p.add_argument(
        "--transfer",
        type=Path,
        default=None,
        help="magnitude_transfer.ecsv from calibrate (Eddington boost estimate)",
    )
    args = p.parse_args(argv)
    if args.outdir is not None:
        args.out_dir = args.outdir
    elif args.out_dir is None:
        args.out_dir = NUMBER_COUNTS_RESULT_DIR
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    transfer = _require_file(args.transfer, "magnitude transfer") if args.transfer else None
    run_number_counts(
        _require_file(args.mosaic or default_mosaic_path(), "mosaic"),
        _require_file(args.catalogue, "catalogue (run classifier.py first)"),
        _require_file(args.completeness, "completeness (run calibrate.py first)"),
        _require_file(args.star_model, "star count model (run classifier.py first)"),
        Path(args.out_dir),
        mask_path=args.mask,
        transfer=transfer,
    )


if __name__ == "__main__":
    main()