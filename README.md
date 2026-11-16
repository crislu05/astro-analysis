# astro-analysis

Year 3 lab cycle 2 astro image processing.

## Layout

FITS inputs live next to this repo, not inside it:

```text
astro/
  astro-analysis/   # this repository
  data/             # e.g. mosaic.fits (local only)
```

Pipeline outputs go under `astro-analysis/result/` (gitignored). Mask inspection PNGs and `summary.json` go to `result/mask/` (`paths.MASK_PLOT_DIR`). Default input: `../data/mosaic.fits` (`paths.DEFAULT_MOSAIC`).

## Run the mask pipeline

From the **repository root** (`astro-analysis/`), with the venv activated:

```powershell
cd D:\astro\astro-analysis
.\.venv\Scripts\Activate.ps1
python mask.py
```

Equivalent:

```powershell
python -m pipeline.mask
```

Optional arguments:

```powershell
python mask.py D:\astro\data\mosaic.fits
python mask.py --no-plots          # mask only, no PNGs
python mask.py -o result\mask      # custom output directory
```

**Do not** run `pipeline/mask/pipeline.py` directly. That file is the mask *builder* module; Python treats the name `pipeline` as the script, so package imports fail.

Programmatic use:

```python
from pathlib import Path
from pipeline.mask.io import run_mosaic_mask

mask, state = run_mosaic_mask(Path("../data/mosaic.fits"))
```