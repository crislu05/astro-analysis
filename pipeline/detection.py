from matplotlib.cm import plasma_r
from scipy import ndimage
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from photutils.utils import circular_footprint
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

data_path = ROOT / 'data'
result_path = ROOT / 'result'

data, header = fits.getdata(data_path / 'mosaic.fits', header=True)

with open(data_path / 'mosaic.txt', "w") as f:
    for card in header.cards:
        f.write(str(card).rstrip() + "\n")


satur = 0.9 * header['SATURATE'] #masks pixels taht are close to saturation as A CCD's response drifts away from linearity.
bad = ~np.isfinite(data)
sat = data > satur

sat_grown = ndimage.binary_dilation(sat, structure=circular_footprint(10))
bleed = ndimage.binary_dilation(sat, structure=np.ones((61,5),bool))


edge = np.zeros_like(bad)
edge[:150,:] = edge[-150:, :] = edge[:, :150] = edge[:, -150:] = True 


yy,xx = np.indices(data.shape)


mask = bad | sat_grown | bleed | edge

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(data, origin='lower', cmap='gray')

overlay = np.zeros((*data.shape,4), dtype=float)
overlay[mask] = [1,0,0,0.35]
ax.imshow(overlay, origin='lower')

plt.tight_layout()
plt.savefig(result_path / 'mosaic_mask.png')


vmin, vmax = ZScaleInterval().get_limits(data)
fig2, ax2 = plt.subplots(figsize=(10,10))
ax2.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
ax2.imshow(overlay, origin='lower')
plt.tight_layout()
plt.savefig(result_path / 'mosaic_zscale.png')


good = ~mask

pixels = data[good].ravel()


fig3, ax3 = plt.subplots(figsize=(10,10))


lo,hi = np.percentile(pixels, (0.1, 99.9))
bins = np.linspace(lo, hi, 500)
ax3.hist(pixels, bins=bins, histtype='step')
ax3.set_xlim(3000,4000)
plt.tight_layout()
plt.savefig(result_path / 'mosaic_histogram.png')