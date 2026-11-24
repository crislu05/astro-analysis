"""CLI for BC95 stellar population validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metcalfe.stella_population.data_paths import DATA_README, BC95Paths, default_bc95_paths
from metcalfe.stella_population.validate import run_validation


def main(argv: list[str] | None = None) -> None:
    defaults = default_bc95_paths()
    parser = argparse.ArgumentParser(
        description="Metcalfe Component C: BC95 SSP reader and exponential SFR spectra.",
    )
    parser.add_argument("--data-dir", type=Path, help="folder containing BC95 .ised files")
    parser.add_argument("--ssp", type=Path, default=defaults.ssp_ised)
    parser.add_argument("--tau1", type=Path, default=defaults.tau1_ised)
    parser.add_argument("--vega", type=Path, default=defaults.vega_spectrum)
    parser.add_argument("--3color", dest="color_table", type=Path, default=defaults.ssp_3color)
    parser.add_argument("--1color", dest="color1_table", type=Path, default=defaults.ssp_1color)
    parser.add_argument("--filterbin", type=Path, default=defaults.filterbin)
    parser.add_argument(
        "--show-data-help",
        action="store_true",
        help="print which files to download and exit",
    )
    args = parser.parse_args(argv)

    if args.show_data_help:
        print(DATA_README)
        return

    if args.data_dir is not None:
        base = args.data_dir
        paths = BC95Paths(
            ssp_ised=base / "bc95_ssp_sp_1.ised",
            tau1_ised=base / "bc95_tau1_sp_1.ised",
            vega_spectrum=base / "alpha_lyrae_kurucz.txt",
            ssp_3color=base / "bc95_ssp_sp_1.3color",
            ssp_1color=base / "bc95_ssp_sp_1.1color",
            filterbin=base / "FILTERBIN.RES",
        )
    else:
        paths = BC95Paths(
            args.ssp,
            args.tau1,
            args.vega,
            args.color_table,
            args.color1_table,
            args.filterbin,
        )

    missing = paths.missing()
    if missing:
        print("Missing BC95 data files:")
        for line in missing:
            print(f"  {line}")
        print()
        print(DATA_README)
        print("\nRun: python metcalfe_stella_population.py --show-data-help")
        sys.exit(2)

    ok = run_validation(paths)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()