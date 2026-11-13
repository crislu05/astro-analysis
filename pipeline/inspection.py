from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

data_path = ROOT / 'data'
result_path = ROOT / 'result'

data, header = fits.getdata(data_path / 'mosaic.fits', header=True)

fig, ax = plt.subplots(figsize=(10,10))

ax.imshow(data, origin='lower', cmap='gray')


plt.tight_layout()
plt.savefig(result_path / 'mosaic_inspect.png')
