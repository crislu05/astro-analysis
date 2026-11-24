# Progression log

## 2026-11-16

- Split monolithic `mask.py` into `pipeline/mask/` modules: `config`, `sky`, `geometry`, `layers`, `border`, `saturated`, `bleed`, `halo`, `streak`, `state`, `pipeline`, `io`.
- Dropped validation, plotting, and source-detection code from the mask entrypoint; `mask.py` is now a thin CLI over `build_mask_state`.
- Fixed `halo_radius` missing `sigma` argument after refactor.
- Added `scipy` and `scikit-image` to `requirements.txt`; fixed CLI `summary.json` numpy int serialization.
- Added `pipeline/mask/plots.py`; CLI writes `mask_overlay.png`, `mask_layers.png`, `mask_zooms.png`, and `good_pixel_histogram.png`.
- Added `.gitignore` with `result/` (outputs only).
- Fixed `pixel_grid` to return `(xx, yy)` so halo, streak, and circle masks align with image coordinates.

## 2026-11-16 (data layout)

- FITS data moved to `astro/data/` (sibling of `astro-analysis`); removed `data/` from `.gitignore`.
- Added `paths.py` with `DATA_DIR`, `DEFAULT_MOSAIC`, and `RESULT_DIR`.
- Documented layout in `README.md`.

## 2026-11-16 (mask CLI)

- Restored entrypoints: `mask.py` and `python -m pipeline.mask` (`pipeline/mask/cli.py`).

## 2026-11-16 (no streak layer)

- Removed streak detection/masking (`streak.py`), `streak` layer, and scikit-image dependency.

## 2026-11-16 (mask plots)

- Added `paths.MASK_PLOT_DIR` (`result/mask`); `save_inspection_plots` defaults there when `outdir` is omitted.

## 2026-11-16 (detect pipeline)

- Organized detect code under `pipeline/detect/`: `config`, `background`, `segment`, `catalogue`, `synthetic`, `pipeline`, `psf`, `validation`, `plots`, `cli`.
- Added `detect.py` / `python -m pipeline.detect`; outputs default to `result/detect/` (`paths.DETECT_RESULT_DIR`).
- Moved synthetic tests to `tests/test_detect_synthetic.py`; added `pytest` to requirements.
- Removed duplicate root copies (`run_detect.py`, `*(2).py`, etc.).

## 2026-11-16 (photometry pipeline)

- Split `photometry.py` into `pipeline/photometry/`: `config`, `inputs`, `aperture`, `calibration`, `correlation`, `cog`, `synthetic`, `injection`, `plots`, `io`, `pipeline`, `selftest`, `cli`.
- Root `photometry.py` is a thin wrapper; outputs default to `result/photometry/`.
- Added `tests/test_photometry_selftest.py`.

## 2026-11-24 (Metcalfe cosmology)

- Refactored `metcalfe/cosmology/cosmology.py` into `config`, `core`, `reference`, `validate`, `summary`, `plots`, `cli`.
- Entrypoints: `metcalfe_cosmology.py`, `python -m metcalfe.cosmology`; figure under `result/metcalfe/`.
- Added `tests/test_metcalfe_cosmology.py`.

## 2026-11-24 (Metcalfe luminosity)

- Refactored `metcalfe/luminosity/luminosity.py` into `gamma`, `schechter`, `presets`, `fit`, `counts`, `validate`, `summary`, `cli`.
- Entrypoints: `metcalfe_luminosity.py`, `python -m metcalfe.luminosity`; uses `metcalfe.cosmology` for convergence checks.
- Added `tests/test_metcalfe_luminosity.py`.

## 2026-11-24 (Metcalfe stellar population)

- Refactored Component C into `metcalfe/stella_population/`: `ised`, `sfr`, `photometry`, `validate`, `data_paths`, `cli`.
- Entrypoints: `metcalfe_stella_population.py`; BC95 data expected in `data/bc95/`.
- Added `tests/test_metcalfe_stella_population.py` (skipped until data present).
- Moved BC95 `.ised`, `FILTERBIN.RES`, and readme PDF to `data/bc95/`.
- Read `FILTERBIN.RES` with csp.f FILTER0 layout (`nf`, `np(100)`, `r(250,100,2)`).
- Added sections 8–9 validation: B–V/V–K vs `.1color`, τ from photographic b–r, K+E grid (53 checks pass).

## 2026-11-17 (calibration CLI)

- Added `paths.default_mosaic_path()` (checks `../data/mosaic.fits` then sibling `mosaic.fits`), `DEFAULT_SDSS_STARS`, `CALIBRATE_RESULT_DIR`.
- Mask/detect/photometry CLIs use `default_mosaic_path()` when mosaic argument is omitted.
- Initial `claude_iter/Calibrate.py` wiring (later moved to `pipeline/calibrate/`).
- Restored positional `out_dir` (7th argument) on `calibrate.py`; `-o` overrides it.
- `paths.default_sdss_stars_path()` picks `data/Skyserver*.csv` under the repo or sibling `../data/`.

## 2026-11-17 (calibrate package)

- Moved B2 calibration from `claude_iter/Calibrate.py` into `pipeline/calibrate/` (`config`, `zeropoint`, `injection`, `zones`, `stats`, `plots`, `pipeline`, `cli`).
- Root `calibrate.py` and `python -m pipeline.calibrate` mirror mask/detect/photometry entrypoints; removed `claude_iter/`.

## 2026-11-17 (.gitignore)

- Ignore `.venv/`, `data/`, `*.fits`, `out_full/`, Python caches, and common OS/IDE junk (keep `result/` ignored).

## 2026-11-17 (classifier package)

- Moved `star_galaxy.py` into `pipeline/classifier/` (`config`, `measure`, `locus`, `classify`, `star_counts`, `plots`, `pipeline`, `cli`).
- Added `paths.CLASSIFIER_RESULT_DIR`, root `classifier.py`, and `python -m pipeline.classifier`.

## 2026-11-17 (number counts package)

- Refactored `number_counts.py` into `pipeline/counts/` (`config`, `zones`, `completeness`, `fitting`, `plots`, `pipeline`, `cli`).
- `paths.NUMBER_COUNTS_RESULT_DIR`; root `number_counts.py` and `python -m pipeline.counts` entrypoints.
- Reuses `pipeline.classifier.zones.zone_of` and `pipeline.calibrate.config.MATCH_RADIUS`.