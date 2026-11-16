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