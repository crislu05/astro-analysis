"""CLI: Component C C2–C6 (tau fit, K+E grids)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metcalfe.stella_population.component_c.run import run_on_bc95
from metcalfe.stella_population.component_c.selftest import run_selftests


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Metcalfe Component C: C2 to C6.")
    parser.add_argument("--selftest", action="store_true", help="synthetic tests only")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    passed = run_selftests()
    if args.selftest:
        sys.exit(0 if passed else 1)

    try:
        from paths import METCALFE_RESULT_DIR
    except ImportError:
        METCALFE_RESULT_DIR = Path("result/metcalfe")

    real = run_on_bc95(args.output_dir or METCALFE_RESULT_DIR)
    sys.exit(0 if passed and real is not False else 1)


if __name__ == "__main__":
    main()