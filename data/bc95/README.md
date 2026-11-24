# BC95 GISSEL 1995 data (Metcalfe Component C)

Download from the STScI/CDBS **Bruzual & Charlot 1995** atlas (`cdbs/grid/bc95/binary`). Place these files in this folder (binaries stay local; not in git):

| File | Role |
|------|------|
| `bc95_ssp_sp_1.ised` | SSP, Salpeter IMF, solar metallicity |
| `bc95_tau1_sp_1.ised` | BC95 exponential SFR τ = 1 Gyr reference |
| `alpha_lyrae_kurucz.txt` | Kurucz Vega spectrum (wavelength Å, flux; from BC95 package) |
| `bc95_ssp_sp_1.3color` | Text table: log age, Mbol, colours, masses |
| `bc95_ssp_sp_1.1color` | Rest-frame U,B,V,K mags (filters 14, 15, 35) |
| `FILTERBIN.RES` | Binary filters (`csp.f` FILTER0: 200,412 bytes) |
| `filterfrm.res`, `filters.log` | Optional ASCII filter source (same bands as FILTERBIN) |

Then run:

```powershell
python metcalfe_stella_population.py
```

Alternate location: `../data/bc95/` (sibling of `astro-analysis`, same as `mosaic.fits`).