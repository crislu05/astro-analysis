"""Backward-compatible entry; prefer ``python -m metcalfe.cosmology`` or ``metcalfe_cosmology.py``."""

from metcalfe.cosmology.cli import main

if __name__ == "__main__":
    main()