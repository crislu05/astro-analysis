"""CLI: validation and summary tables."""

from __future__ import annotations

import argparse
import sys

from metcalfe.luminosity.summary import print_tables
from metcalfe.luminosity.validate import run_validation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Metcalfe Component B: Schechter luminosity functions.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run checks only (no summary table)",
    )
    parser.add_argument("--no-validate", action="store_true", help="skip validation checks")
    parser.add_argument("--no-table", action="store_true", help="skip summary table")
    args = parser.parse_args(argv)

    all_passed = True
    if not args.no_validate:
        all_passed = run_validation()

    if not args.validate_only and not args.no_table:
        print_tables()

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()