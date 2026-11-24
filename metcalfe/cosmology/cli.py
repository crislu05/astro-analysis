"""CLI: validation, summary table, Component A figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paths import METCALFE_RESULT_DIR
from metcalfe.cosmology.plots import save_component_a_figure
from metcalfe.cosmology.summary import print_summary_table
from metcalfe.cosmology.validate import run_validation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Metcalfe Component A: cosmology validation and Figure 2.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run checks only (no summary table or figure)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip validation checks",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="skip Figure 2 plot",
    )
    parser.add_argument(
        "-o",
        "--figure",
        type=Path,
        default=METCALFE_RESULT_DIR / "component_a_cosmology.png",
        help="output path for Component A figure",
    )
    args = parser.parse_args(argv)

    all_passed = True
    if not args.no_validate:
        all_passed = run_validation()

    if not args.validate_only:
        if not args.no_validate:
            print_summary_table()
        if not args.no_figure:
            save_component_a_figure(args.figure)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()