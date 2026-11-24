"""Synthetic SSP and filters for Component C self-tests."""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from metcalfe.stella_population.component_c.config import Bands, VEGA_FLUX_5556_CGS
from metcalfe.stella_population.ised import StellarPopulation
from metcalfe.stella_population.photometry import Filter


def _planck_lambda(wavelengths_aa, temperature_k):
    wavelength_cm = wavelengths_aa * 1e-8
    exponent = np.minimum(1.4388 / (wavelength_cm * temperature_k), 700.0)
    return 1.0 / (wavelength_cm**5 * np.expm1(exponent))


def synthetic_ssp(red_supergiant_blip=False):
    ages_yr = np.concatenate([[0.0], np.geomspace(1e5, 2e10, 220)])
    wavelengths = np.geomspace(91.0, 1.0e6, 1206)
    giant = _planck_lambda(wavelengths, 4000.0)
    giant /= np.trapezoid(giant, wavelengths)
    spectra = np.zeros((len(ages_yr), len(wavelengths)))
    for index, age in enumerate(ages_yr):
        age_gyr = max(age, 1e5) / 1e9
        turnoff_temperature = 5500.0 + 34500.0 / (1.0 + (age_gyr / 0.005) ** 0.6)
        hot = _planck_lambda(wavelengths, turnoff_temperature)
        hot /= np.trapezoid(hot, wavelengths)
        giant_share = 0.05 + 0.6 * age_gyr / (age_gyr + 0.3)
        if red_supergiant_blip and 0.005 < age_gyr < 0.03:
            giant_share = 0.95
        luminosity = 1.0 / (1.0 + age_gyr / 0.003) ** 0.9
        spectra[index] = luminosity * ((1.0 - giant_share) * hot + giant_share * giant)
    mass = 1.0 - 0.3 * np.log10(1.0 + ages_yr / 1e8) / np.log10(1.0 + 2e10 / 1e8)
    return StellarPopulation(
        "synthetic",
        ages_yr,
        wavelengths,
        spectra,
        np.array([[1.35, 0.1, 125.0, 0, 0, 0]]),
        0,
        0.0,
        extras={"stellar_mass": mass},
    )


def synthetic_bands():
    def top_hat(name, centre, width):
        wavelengths = np.linspace(centre - width / 2, centre + width / 2, 300)
        return Filter(name, wavelengths, np.ones_like(wavelengths))

    vega_wavelengths = np.geomspace(500.0, 3.0e5, 6000)
    vega_flux = _planck_lambda(vega_wavelengths, 9600.0)
    vega_flux *= VEGA_FLUX_5556_CGS / np.interp(5556.0, vega_wavelengths, vega_flux)
    vega = (vega_wavelengths, vega_flux)
    return Bands(top_hat("b", 4600.0, 1000.0), top_hat("r", 6400.0, 1000.0), vega), top_hat(
        "B", 4400.0, 900.0
    )