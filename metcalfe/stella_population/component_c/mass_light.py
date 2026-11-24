"""C4 extras: Vega units check, absolute magnitudes, M/L_B."""

from __future__ import annotations

import numpy as np

from metcalfe.stella_population import config as sp_config
from metcalfe.stella_population.component_c.config import (
    PARSEC_CM,
    PHOTON_COUNTING,
    SUN_ABSOLUTE_B_VEGA,
    VEGA_FLUX_5556_CGS,
)
from metcalfe.stella_population.component_c.spectrum import stellar_mass
from metcalfe.stella_population.photometry import Filter, band_flux
from metcalfe.stella_population.sfr import exponential_sfr_spectrum


def vega_units_look_physical(vega):
    wavelengths, flux = vega
    value = float(np.interp(5556.0, wavelengths, flux))
    return 0.5 < value / VEGA_FLUX_5556_CGS < 2.0


def absolute_magnitude(wavelengths_aa, spectrum_lsun_per_aa, band: Filter, vega):
    flux = spectrum_lsun_per_aa * sp_config.SOLAR_LUMINOSITY_ERG_S / (
        4.0 * np.pi * (10.0 * PARSEC_CM) ** 2
    )
    vega_wavelengths, vega_flux = vega
    return float(
        -2.5
        * np.log10(
            band_flux(wavelengths_aa, flux, band, 0.0, PHOTON_COUNTING)
            / band_flux(vega_wavelengths, vega_flux, band, 0.0, PHOTON_COUNTING)
        )
    )


def mass_to_light_b(ssp, tau_gyr, age_gyr, band_b, vega):
    spectrum = exponential_sfr_spectrum(ssp, tau_gyr, age_gyr)
    magnitude_b = absolute_magnitude(ssp.wavelengths_aa, spectrum, band_b, vega)
    luminosity_b_solar = 10.0 ** (-0.4 * (magnitude_b - SUN_ABSOLUTE_B_VEGA))
    return stellar_mass(ssp, tau_gyr, age_gyr) / luminosity_b_solar