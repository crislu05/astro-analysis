"""Command-line entry for star/galaxy classification."""

from __future__ import annotations

import argparse
from pathlib import Path

from paths import (
    default_mosaic_path,
    default_sdss_stars_path,
    default_segmentation_path,
    pipeline_result_dir,
)
from pipeline.classifier.pipeline import run_classifier


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Star/galaxy separation via concentration and the stellar locus.",
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
        help="calibrated catalogue with r_point (default: result/calibrate/<band>/...)",
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
        "sdss_stars",
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
        help="Output directory (default: result/classifier/<band>)",
    )
    p.add_argument("-o", "--outdir", type=Path, default=None, help="output directory")
    p.add_argument(
        "--sdss-galaxies",
        type=Path,
        default=None,
        help="optional SDSS galaxy table for purity check",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mosaic = args.mosaic or default_mosaic_path()
    detect_dir = pipeline_result_dir("detect", mosaic)
    args.catalogue = args.catalogue or (
        pipeline_result_dir("calibrate", mosaic) / "calibrated_catalogue.ecsv"
    )
    args.segmentation = args.segmentation or default_segmentation_path(detect_dir)
    if args.outdir is not None:
        args.out_dir = args.outdir
    elif args.out_dir is None:
        args.out_dir = pipeline_result_dir("classifier", mosaic)
    galaxies = _require_file(args.sdss_galaxies, "SDSS galaxies") if args.sdss_galaxies else None
    run_classifier(
        _require_file(mosaic, "mosaic"),
        _require_file(args.catalogue, "catalogue (run calibrate.py first)"),
        _require_file(args.segmentation, "segmentation"),
        _require_file(args.sdss_stars or default_sdss_stars_path(), "SDSS stars CSV"),
        Path(args.out_dir),
        mask_path=args.mask,
        sdss_galaxies=galaxies,
    )


if __name__ == "__main__":
    main()