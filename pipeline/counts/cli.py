"""Command-line entry for galaxy number counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from paths import default_mosaic_path, pipeline_result_dir
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
        default=None,
    )
    p.add_argument(
        "completeness",
        nargs="?",
        type=Path,
        default=None,
    )
    p.add_argument(
        "star_model",
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
        "out_dir",
        nargs="?",
        type=Path,
        default=None,
        metavar="out_dir",
        help="Output directory (default: result/number_counts/<band>)",
    )
    p.add_argument("-o", "--outdir", type=Path, default=None)
    p.add_argument(
        "--transfer",
        type=Path,
        default=None,
        help="magnitude_transfer.ecsv from calibrate (Eddington boost estimate)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mosaic = args.mosaic or default_mosaic_path()
    cal_dir = pipeline_result_dir("calibrate", mosaic)
    cls_dir = pipeline_result_dir("classifier", mosaic)
    args.catalogue = args.catalogue or (cls_dir / "classified_catalogue.ecsv")
    args.completeness = args.completeness or (cal_dir / "completeness_galaxies.ecsv")
    args.star_model = args.star_model or (cls_dir / "star_count_model.ecsv")
    if args.outdir is not None:
        args.out_dir = args.outdir
    elif args.out_dir is None:
        args.out_dir = pipeline_result_dir("number_counts", mosaic)
    transfer = _require_file(args.transfer, "magnitude transfer") if args.transfer else None
    run_number_counts(
        _require_file(mosaic, "mosaic"),
        _require_file(args.catalogue, "catalogue (run classifier.py first)"),
        _require_file(args.completeness, "completeness (run calibrate.py first)"),
        _require_file(args.star_model, "star count model (run classifier.py first)"),
        Path(args.out_dir),
        mask_path=args.mask,
        transfer=transfer,
    )


if __name__ == "__main__":
    main()