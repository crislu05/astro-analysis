"""Repository-adjacent paths (FITS data is not tracked in git)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ASTRO_ROOT = REPO_ROOT.parent

DATA_DIR = ASTRO_ROOT / "data"
REPO_DATA_DIR = REPO_ROOT / "data"
RESULT_DIR = REPO_ROOT / "result"
MASK_PLOT_DIR = RESULT_DIR / "mask"
DETECT_RESULT_DIR = RESULT_DIR / "detect"
PHOTOMETRY_RESULT_DIR = RESULT_DIR / "photometry"

DEFAULT_MOSAIC = DATA_DIR / "mosaic.fits"
DEFAULT_SDSS_STARS = DATA_DIR / "sdss_stars.csv"
CALIBRATE_RESULT_DIR = RESULT_DIR / "calibrate"


def default_mosaic_path() -> Path:
    """First existing mosaic among standard lab locations."""
    for candidate in (
        REPO_DATA_DIR / "mosaic.fits",
        DATA_DIR / "mosaic.fits",
        ASTRO_ROOT / "mosaic.fits",
    ):
        if candidate.is_file():
            return candidate
    return DEFAULT_MOSAIC


def default_sdss_stars_path() -> Path:
    """SDSS calibration star list (CSV from SkyServer / lab)."""
    for candidate in (
        REPO_DATA_DIR / "sdss_stars.csv",
        DATA_DIR / "sdss_stars.csv",
    ):
        if candidate.is_file():
            return candidate
    for folder in (REPO_DATA_DIR, DATA_DIR):
        if not folder.is_dir():
            continue
        matches = sorted(folder.glob("Skyserver*.csv"))
        if matches:
            return matches[0]
    return DEFAULT_SDSS_STARS