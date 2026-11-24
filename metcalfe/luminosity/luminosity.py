"""Backward-compatible entry; prefer ``metcalfe_luminosity.py`` or ``python -m metcalfe.luminosity``."""

from metcalfe.luminosity.cli import main

if __name__ == "__main__":
    main()