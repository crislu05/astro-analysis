"""Command-line entry for aperture photometry."""

from __future__ import annotations

import argparse
from pathlib import Path

from paths import DETECT_RESULT_DIR, PHOTOMETRY_RESULT_DIR, default_mosaic_path
from pipeline.photometry.io import load_detection_inputs, load_mask
from pipeline.photometry.inputs import PhotometryInputs
from pipeline.photometry.pipeline import run_photometry_catalogue
from pipeline.photometry.selftest import selftest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aperture photometry from detection products (lab B2 §5.4–5.6).",
    )
    parser.add_argument(
        "mosaic",
        nargs="?",
        type=Path,
        default=None,
        help=f"FITS mosaic (default: {default_mosaic_path()})",
    )
    parser.add_argument(
        "catalogue",
        nargs="?",
        type=Path,
        default=DETECT_RESULT_DIR / "catalogue.ecsv",
    )
    parser.add_argument(
        "segmentation",
        nargs="?",
        type=Path,
        default=DETECT_RESULT_DIR / "segmentation.fits",
    )
    parser.add_argument(
        "mask",
        nargs="?",
        type=Path,
        default=None,
        help="mask .fits/.npy; if omitted, rebuild from mosaic via mask pipeline",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=PHOTOMETRY_RESULT_DIR,
        help=f"Output directory (default: {PHOTOMETRY_RESULT_DIR})",
    )
    parser.add_argument("--selftest", action="store_true", help="run synthetic self-tests")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="inject fake stars to test bias and errors",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        selftest()
        return

    args.mosaic = args.mosaic or default_mosaic_path()

    for label, path in (
        ("mosaic", args.mosaic),
        ("catalogue", args.catalogue),
        ("segmentation", args.segmentation),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    mask_array = load_mask(args.mask) if args.mask is not None else None
    data, header, cat, segm, mask, gain = load_detection_inputs(
        args.mosaic, args.catalogue, args.segmentation, mask=mask_array
    )
    zp = float(header["MAGZPT"])
    zp_err = float(header.get("MAGZRR", 0.0))
    exptime = float(header.get("EXPTIME", 720.0))
    print(f"{len(cat)} sources, gain {gain}, EXPTIME {exptime}, MAGZPT {zp} +- {zp_err}")

    inp = PhotometryInputs(data, mask, segm, gain)
    run_photometry_catalogue(
        inp,
        cat,
        zp=zp,
        zp_err=zp_err,
        exptime=exptime,
        out=args.outdir,
        validate=args.validate,
    )
    print(f"wrote {args.outdir / 'photometry.ecsv'}")


if __name__ == "__main__":
    main()