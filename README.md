# astro-analysis

Year 3 lab cycle 2 astro image processing.

## Layout

```text
astro/
  astro-analysis/     # this repository
  data/               # mosaic_r.fits + other bands (local only)

astro-analysis/
  pipeline/
    mask/             # artefact masking
    detect/           # source detection + validation
    photometry/       # aperture photometry
    bands/            # cut multi-band SWIRE tiles onto the lab grid
    calibrate/        # SDSS zero point + galaxy calibration
    classifier/       # star/galaxy separation
    counts/           # differential galaxy number counts
  metcalfe/           # Metcalfe (2001) count model checks (cosmology, …)
  tests/              # pytest (synthetic detect + photometry tests)
  mask.py             # run mask CLI
  detect.py           # run detect + validation CLI
  photometry.py       # run photometry CLI
  bands.py            # extract i, U, g mosaics onto lab grid
  calibrate.py        # run photometric calibration CLI
  classifier.py       # run star/galaxy classifier CLI
  result/             # gitignored outputs
      mask/
      detect/
      photometry/
      calibrate/
      classifier/
      number_counts/
      metcalfe/
```

FITS default: `../data/mosaic_r.fits` (`paths.default_mosaic_path()`; accepts legacy `mosaic.fits`).

## Mask pipeline

From the repo root with venv activated:

```powershell
python mask.py
python -m pipeline.mask
```

Outputs: `result/mask/` (PNGs + `summary.json`).

## Multi-band mosaics

Put the lab **r** mosaic as `data/mosaic_r.fits` and the SWIRE archive tiles in the **same folder**. Extraction writes `mosaic_i.fits`, `mosaic_u.fits`, `mosaic_g.fits` on the **same** WCS and header layout (any band file works with mask, detect, photometry):

```powershell
python bands.py
python -m pipeline.bands
python mask.py data/mosaic_g.fits
python detect.py D:\astro\data\mosaic_g.fits
```

All **FITS** (mosaics, segmentation, band coverage) live under **`../data/`**. JSON/ECSV/PNGs stay in **`result/`** (e.g. `result/bands/extraction_report.json`).

## Detect pipeline

```powershell
python detect.py
python -m pipeline.detect
```

Outputs: `result/detect/<band>/` (e.g. `r`, `i`, `U`, `g`) — `catalogue.ecsv`, `validation.json`, PNGs; segmentation FITS → `../data/pipeline/result/detect/<band>/`.

```powershell
python detect.py D:\astro\data\mosaic_r.fits -o result\detect
```

## Photometry pipeline

Uses matching `result/detect/<band>/` products by default; rebuilds the mask from the mosaic unless you pass a mask file.

```powershell
python photometry.py --selftest
python photometry.py
python -m pipeline.photometry
python photometry.py --validate    # also run PSF injection checks (slow)
```

Outputs: `result/photometry/` — `photometry.ecsv`, `photometry_summary.json`, `photometry_diagnostics.png`.

## Calibrate pipeline

Uses detect + photometry outputs and an SDSS star CSV from `data/` (SkyServer export).

```powershell
python calibrate.py
python calibrate.py --quick
python -m pipeline.calibrate
```

Outputs: `result/calibrate/` — `calibrated_catalogue.ecsv`, `completeness_galaxies.ecsv`, `magnitude_transfer.ecsv`, `calibration_summary.json`, diagnostic PNGs.

## Classifier pipeline

Uses `result/calibrate/calibrated_catalogue.ecsv` (needs `r_point`), segmentation, and SDSS stars.

```powershell
python classifier.py
python -m pipeline.classifier
python classifier.py --sdss-galaxies data/galaxies.csv
```

Outputs: `result/classifier/` — `classified_catalogue.ecsv`, `star_count_model.ecsv`, `stellar_locus.ecsv`, `star_galaxy_summary.json`, `star_galaxy.png`.

## Number counts

Uses classified catalogue, calibrate completeness, and classifier star-count model.

```powershell
python number_counts.py
python -m pipeline.counts
python number_counts.py --transfer result/calibrate/magnitude_transfer.ecsv
```

Outputs: `result/number_counts/` — `number_counts.ecsv`, `number_counts_fit.json`, `number_counts.png`.

## Metcalfe stellar population (Component C)

BC95 `.ised` SSP files (not in git). See `data/bc95/README.md`.

**BC95 reader validation** (SSP, filters, photometry checks):

```powershell
python metcalfe_stella_population.py --show-data-help
python metcalfe_stella_population.py
python -m metcalfe.stella_population
```

**C2–C6** (tau fit, legacy regression, [K+E] grids):

```powershell
python metcalfe_component_c_fit.py --selftest
python metcalfe_component_c_fit.py
python -m metcalfe.stella_population.component_c
```

Package layout: `metcalfe/stella_population/` (reader, `validate.py`) and `metcalfe/stella_population/component_c/` (tau, K+E, `run.py`).

Outputs: `result/metcalfe/component_c_tau.json`, `component_c_kegrid.npz`, `component_c_evolution.png`.

## Metcalfe luminosity (Component B)

Five Schechter LFs, h rescaling, normalisation fit A, validation suite.

```powershell
python metcalfe_luminosity.py
python -m metcalfe.luminosity
```

## Metcalfe cosmology (Component A)

Legacy vs Planck cosmologies, validation suite, and proposal Figure 2.

```powershell
python metcalfe_cosmology.py
python -m metcalfe.cosmology
python metcalfe_cosmology.py --validate-only
```

Outputs: `result/metcalfe/component_a_cosmology.png` (and validation printed to the console).

## Tests

```powershell
python -m pytest tests/ -v
```

Do **not** run `pipeline/mask/pipeline.py`, `pipeline/detect/pipeline.py`, `pipeline/calibrate/pipeline.py`, `pipeline/classifier/pipeline.py`, or `pipeline/counts/pipeline.py` as scripts (name clashes with the `pipeline` package).

## Programmatic use

```python
from pipeline.mask import build_mask_state
from pipeline.detect import detect

state = build_mask_state(data, header)
result = detect(state.data, state.mask, gain=3.1)
```