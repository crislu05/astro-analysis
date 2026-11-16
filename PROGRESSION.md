# Progression log

## 2026-11-16

- Split monolithic `mask.py` into `pipeline/mask/` modules: `config`, `sky`, `geometry`, `layers`, `border`, `saturated`, `bleed`, `halo`, `streak`, `state`, `pipeline`, `io`.
- Dropped validation, plotting, and source-detection code from the mask entrypoint; `mask.py` is now a thin CLI over `build_mask_state`.
- Fixed `halo_radius` missing `sigma` argument after refactor.
- Added `scipy` and `scikit-image` to `requirements.txt`; fixed CLI `summary.json` numpy int serialization.
- Added `pipeline/mask/plots.py`; CLI writes `mask_overlay.png`, `mask_layers.png`, `mask_zooms.png`, and `good_pixel_histogram.png`.
- Added `.gitignore` with `result/` and `data/`.