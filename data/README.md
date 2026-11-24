# Local data (not tracked in git except these README files)

Large inputs live outside version control. Pipelines search **`data/`** inside this repo and **`../data/`** next to it (`paths.DATA_DIR` and `paths.REPO_DATA_DIR`).

| Location | Typical contents |
|----------|------------------|
| `data/mosaic.fits` or `../data/mosaic.fits` | Lockman mosaic for mask / detect / photometry |
| `data/sdss_stars.csv` | SkyServer star list for calibration |
| `data/bc95/` | Bruzual & Charlot 1995 files for Metcalfe Component C (see `bc95/README.md`) |

Clone the repo, then copy or download datasets into the paths above before running the matching CLIs.