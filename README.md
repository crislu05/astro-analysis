# astro-analysis

Year 3 lab cycle 2 astro image processing.

## Layout

```text
astro/
  astro-analysis/     # this repository
  data/               # mosaic.fits (local only)

astro-analysis/
  pipeline/
    mask/             # artefact masking
    detect/           # source detection + validation
    photometry/       # aperture photometry + calibration
  tests/              # pytest (synthetic detect + photometry tests)
  mask.py             # run mask CLI
  detect.py           # run detect + validation CLI
  photometry.py       # run photometry CLI
  result/             # gitignored outputs
      mask/
      detect/
      photometry/
```

FITS default: `../data/mosaic.fits` (`paths.DEFAULT_MOSAIC`).

## Mask pipeline

From the repo root with venv activated:

```powershell
python mask.py
python -m pipeline.mask
```

Outputs: `result/mask/` (PNGs + `summary.json`).

## Detect pipeline

```powershell
python detect.py
python -m pipeline.detect
```

Outputs: `result/detect/` — `catalogue.ecsv`, `segmentation.fits`, `validation.json`, validation PNGs.

```powershell
python detect.py D:\astro\data\mosaic.fits -o result\detect
```

## Photometry pipeline

Uses `result/detect/catalogue.ecsv` and `segmentation.fits` by default; rebuilds the mask from the mosaic unless you pass a mask file.

```powershell
python photometry.py --selftest
python photometry.py
python -m pipeline.photometry
python photometry.py --validate    # also run PSF injection checks (slow)
```

Outputs: `result/photometry/` — `photometry.ecsv`, `photometry_summary.json`, `photometry_diagnostics.png`.

## Tests

```powershell
python -m pytest tests/ -v
```

Do **not** run `pipeline/mask/pipeline.py` or `pipeline/detect/pipeline.py` as scripts (name clashes with the `pipeline` package).

## Programmatic use

```python
from pipeline.mask import build_mask_state
from pipeline.detect import detect

state = build_mask_state(data, header)
result = detect(state.data, state.mask, gain=3.1)
```