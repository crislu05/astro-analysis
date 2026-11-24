"""CLI: cut SWIRE tiles onto the lab mosaic grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paths import (
    BANDS_RESULT_DIR,
    band_from_mosaic_path,
    band_mosaic_path,
    default_mosaic_path,
    mosaic_data_dir,
    mosaic_product_dir,
)
from pipeline.bands.config import DEFAULT_BANDS, DEFAULT_REFERENCE_BAND, DEFAULT_TILES
from pipeline.bands.pipeline import ExtractConfig, extract_bands


def main(argv: list[str] | None = None) -> None:
    lab_default = default_mosaic_path()
    data_default = mosaic_data_dir(lab_default)
    parser = argparse.ArgumentParser(
        description=(
            "Resample SWIRE/KPNO archive tiles onto the lab mosaic grid. "
            "FITS under ../data; JSON report under result/bands/."
        ),
    )
    parser.add_argument(
        "--lab-mosaic",
        type=Path,
        default=None,
        help=f"Reference r-band lab mosaic (default: {lab_default})",
    )
    parser.add_argument(
        "--tile-dir",
        type=Path,
        default=None,
        help=f"Folder with LH_Mosaic_opt_tile_*_*_mos32.fits (default: same folder as lab mosaic: {data_default})",
    )
    parser.add_argument(
        "--tiles",
        nargs="+",
        default=list(DEFAULT_TILES),
        help="Tile ids to combine (first listed has priority in overlap)",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=list(DEFAULT_BANDS),
        help="Bands to extract (archive U filter uses uppercase in filenames)",
    )
    parser.add_argument(
        "-o",
        "--report-dir",
        type=Path,
        default=BANDS_RESULT_DIR,
        help=f"JSON extraction report (default: {BANDS_RESULT_DIR}); coverage FITS go to ../data/bands/",
    )
    args = parser.parse_args(argv)

    lab = args.lab_mosaic or lab_default
    tile_dir = args.tile_dir or mosaic_data_dir(lab)
    if not lab.is_file():
        raise SystemExit(f"lab mosaic not found: {lab}")
    if not tile_dir.is_dir():
        raise SystemExit(
            f"data folder not found: {tile_dir}\n"
            f"Put mosaic_r.fits (lab r band) and LH_Mosaic_opt_tile_*_*_mos32.fits in the same directory."
        )

    print(f"Lab mosaic: {lab}")
    print(f"Tiles:      {tile_dir}")
    product_dir = mosaic_product_dir(lab)
    print(f"Outputs:    {product_dir}/mosaic_<band>.fits")
    print(f"Report:     {args.report_dir}  (coverage FITS: ../data/bands/)")
    config = ExtractConfig(
        lab_mosaic=lab,
        tile_dir=tile_dir,
        tiles=tuple(args.tiles),
        bands=tuple(args.bands),
        reference_band=DEFAULT_REFERENCE_BAND,
        report_dir=args.report_dir,
    )
    extract_bands(config)
    written = [band_mosaic_path(b, lab) for b in args.bands if band_mosaic_path(b, lab).is_file()]
    if written:
        print("\nRun downstream pipelines per band, e.g.:")
        for path in written:
            band = band_from_mosaic_path(path)
            print(f"  python mask.py {path}")
            print(f"  python detect.py {path}")
            print(f"    -> result/mask/{band}/  result/detect/{band}/")
    sys.exit(0)


if __name__ == "__main__":
    main()