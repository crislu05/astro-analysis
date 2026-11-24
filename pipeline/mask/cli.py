"""Command-line entry for the mosaic mask pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paths import default_mosaic_path, pipeline_result_dir
from pipeline.mask.config import LAYER_NAMES
from pipeline.mask.io import run_mosaic_mask
from pipeline.mask.plots import save_inspection_plots
from pipeline.mask.state import MaskState


def write_summary(state: MaskState, mosaic: Path, outdir: Path) -> Path:
    n_pix = int(state.data.size)
    layers = state.layers
    summary = {
        "mosaic": str(mosaic.resolve()),
        "shape": [int(state.data.shape[0]), int(state.data.shape[1])],
        "sky": float(state.sky),
        "sigma": float(state.sigma),
        "saturation": float(state.saturation),
        "masked_pixels": int(state.mask.sum()),
        "masked_fraction": float(state.mask.sum() / n_pix),
        "layers": {
            name: int(getattr(layers, name).sum()) for name in LAYER_NAMES
        },
        "n_bleeds": len(state.bleeds),
        "n_stars": len(state.stars),
    }
    path = outdir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the multi-layer artefact mask for the KPNO mosaic.",
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
        default=None,
        help="Inspection plots and summary.json (default: result/mask/<band>)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG inspection figures",
    )
    args = parser.parse_args(argv)
    args.mosaic = args.mosaic or default_mosaic_path()

    if not args.mosaic.is_file():
        raise SystemExit(f"mosaic not found: {args.mosaic}")

    print(f"Masking {args.mosaic} ...", flush=True)
    mask, state = run_mosaic_mask(args.mosaic)
    print(
        f"Done: {int(mask.sum()):,} / {mask.size:,} pixels masked "
        f"({100 * mask.mean():.2f}%)",
        flush=True,
    )

    if args.no_plots:
        return

    plot_dir = args.outdir or pipeline_result_dir("mask", args.mosaic)
    outdir = save_inspection_plots(state, plot_dir)
    summary = write_summary(state, args.mosaic, outdir)
    print(f"Wrote plots to {outdir}")
    print(f"Wrote {summary.name}")


if __name__ == "__main__":
    main()