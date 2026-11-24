# Local data (not tracked in git except these README files)

**No FITS files belong inside the repo.** Inputs, mosaics, segmentation maps, and band-coverage FITS go in **`../data/`** (`paths.DATA_DIR`). The repo’s `data/` folder is README-only; pipeline tables and plots stay in **`result/`**.

| Location | Typical contents |
|----------|------------------|
| `data/mosaic_r.fits` | Lockman **r**-band lab mosaic — this *is* the r band on the reference grid (not rebuilt from tiles). Archive `*_r_*` tiles are optional. Legacy name `mosaic.fits` still works. |
| `../data/mosaic_{i,u,g,...}.fits` | Bands i, U, g cut onto the same grid (`python bands.py`) |
| `../data/bands/coverage_*.fits` | Band extraction coverage maps |
| `../data/pipeline/` | Pipeline FITS mirrored from would-be repo paths (e.g. segmentation) |
| `result/bands/` | Extraction `extraction_report.json` only |
| `data/LH_Mosaic_opt_tile_*_*_mos32.fits` | SWIRE archive tiles (same folder as `mosaic_r.fits`) |
| `data/sdss_stars.csv` | SkyServer star list for calibration |
| `data/bc95/` | Bruzual & Charlot 1995 files for Metcalfe Component C (see `bc95/README.md`) |

Clone the repo, then copy or download datasets into the paths above before running the matching CLIs.