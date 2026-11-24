"""SWIRE tile extraction defaults."""

from pathlib import Path

TILE_PATTERN = "LH_Mosaic_opt_tile_{tile}_{band}_mos32.fits"
DEFAULT_TILES = ("4_4", "4_3")
DEFAULT_REFERENCE_BAND = "r"
DEFAULT_BANDS = ("i", "u", "g")  # archive filenames use U for the u band
FILL_VALUE = 0.0
MAX_OFFSET_RESIDUAL_PX = 1.0

BLOCK_ROWS = 128
PATCH_HALF = 128
SEARCH_RADIUS = 25
CALIBRATION_KEYS = (
    "FILTER",
    "MAGZPT",
    "MAGZRR",
    "EXPTIME",
    "GAIN",
    "SATURATE",
    "BUNIT",
)


def archive_band_name(band: str) -> str:
    """Band string in archive filenames (U is uppercase)."""
    return "U" if band.lower() == "u" else band.lower()


def tile_path(tile_dir: Path, tile: str, band: str) -> Path:
    return tile_dir / TILE_PATTERN.format(tile=tile, band=archive_band_name(band))