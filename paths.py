"""Repository-adjacent paths (FITS data is not tracked in git)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ASTRO_ROOT = REPO_ROOT.parent

DATA_DIR = ASTRO_ROOT / "data"
RESULT_DIR = REPO_ROOT / "result"
MASK_PLOT_DIR = RESULT_DIR / "mask"

DEFAULT_MOSAIC = DATA_DIR / "mosaic.fits"