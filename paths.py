"""Repository paths: FITS only under sibling ``../data``; tables/plots under ``result/``."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ASTRO_ROOT = REPO_ROOT.parent

DATA_DIR = ASTRO_ROOT / "data"
REPO_DATA_DIR = REPO_ROOT / "data"
RESULT_DIR = REPO_ROOT / "result"
MASK_PLOT_DIR = RESULT_DIR / "mask"
DETECT_RESULT_DIR = RESULT_DIR / "detect"
PHOTOMETRY_RESULT_DIR = RESULT_DIR / "photometry"
METCALFE_RESULT_DIR = RESULT_DIR / "metcalfe"

REFERENCE_R_BAND = "r"
LAB_MOSAIC_FILENAMES = ("mosaic_r.fits", "mosaic.fits")  # legacy second name
DEFAULT_MOSAIC = DATA_DIR / LAB_MOSAIC_FILENAMES[0]
DEFAULT_SDSS_STARS = DATA_DIR / "sdss_stars.csv"
CALIBRATE_RESULT_DIR = RESULT_DIR / "calibrate"
CLASSIFIER_RESULT_DIR = RESULT_DIR / "classifier"
NUMBER_COUNTS_RESULT_DIR = RESULT_DIR / "number_counts"
BANDS_RESULT_DIR = RESULT_DIR / "bands"

BAND_MOSAIC_PREFIX = "mosaic_"
FITS_DATA_ROOT = DATA_DIR / "pipeline"
DATA_BANDS_FITS_DIR = DATA_DIR / "bands"


def _path_under_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
        return True
    except ValueError:
        return False


def mosaic_product_dir(lab_mosaic: Path | None = None) -> Path:
    """FITS mosaics and band products — prefer sibling ``../data``, not files inside the repo."""
    lab = lab_mosaic or default_mosaic_path()
    parent = lab.parent.resolve()
    if _path_under_repo(parent):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return DATA_DIR
    return parent


def fits_product_path(path: Path) -> Path:
    """Map a FITS path under the repo to ``../data/pipeline/<same relative path>``."""
    path = Path(path)
    if not _path_under_repo(path):
        return path
    rel = path.resolve().relative_to(REPO_ROOT.resolve())
    dest = FITS_DATA_ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def default_segmentation_path(
    detect_outdir: Path | None = None,
    mosaic: Path | None = None,
) -> Path:
    """Segmentation map FITS for a detect run (never under the repo tree)."""
    outdir = detect_outdir or pipeline_result_dir("detect", mosaic)
    return fits_product_path(outdir / "segmentation.fits")


def band_coverage_path(band: str) -> Path:
    """Per-band coverage map from the bands extractor."""
    DATA_BANDS_FITS_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_BANDS_FITS_DIR / f"coverage_{band.lower()}.fits"


def band_from_mosaic_path(mosaic: Path) -> str:
    """Band subfolder name under ``result/<stage>/`` (archive u → ``U``)."""
    name = mosaic.name.lower()
    legacy = {n.lower() for n in LAB_MOSAIC_FILENAMES}
    if name in legacy:
        return REFERENCE_R_BAND
    prefix = BAND_MOSAIC_PREFIX.lower()
    if name.startswith(prefix) and name.endswith(".fits"):
        band = name[len(prefix) : -5]
        return "U" if band == "u" else band
    return REFERENCE_R_BAND


def pipeline_result_dir(stage: str, mosaic: Path | None = None) -> Path:
    """Per-band outputs, e.g. ``result/detect/r``, ``result/detect/U``."""
    mosaic = mosaic or default_mosaic_path()
    return RESULT_DIR / stage / band_from_mosaic_path(mosaic)


def default_mosaic_path() -> Path:
    """Lab r-band mosaic (`mosaic_r.fits`, or legacy `mosaic.fits`)."""
    for parent in (DATA_DIR, ASTRO_ROOT):
        for name in LAB_MOSAIC_FILENAMES:
            candidate = parent / name
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


def mosaic_data_dir(lab_mosaic: Path | None = None) -> Path:
    """Folder that holds mosaic_r.fits and the SWIRE archive tiles (same directory)."""
    lab = lab_mosaic or default_mosaic_path()
    return lab.parent


def resolve_swire_tile_dir(lab_mosaic: Path | None = None) -> Path:
    """Alias for mosaic_data_dir (tiles live beside the lab mosaic)."""
    return mosaic_data_dir(lab_mosaic)


def band_mosaic_path(band: str, lab_mosaic: Path | None = None) -> Path:
    """FITS on the lab pixel grid for a given filter (e.g. mosaic_g.fits)."""
    return mosaic_product_dir(lab_mosaic) / f"{BAND_MOSAIC_PREFIX}{band.lower()}.fits"


def list_band_mosaics(lab_mosaic: Path | None = None) -> list[Path]:
    """Band mosaics in the external data folder, excluding the reference r file itself."""
    lab = lab_mosaic or default_mosaic_path()
    parent = mosaic_product_dir(lab)
    if not parent.is_dir():
        return []
    legacy_names = set(LAB_MOSAIC_FILENAMES)
    return sorted(
        p
        for p in parent.glob(f"{BAND_MOSAIC_PREFIX}*.fits")
        if p.name not in legacy_names and p.name != lab.name
    )