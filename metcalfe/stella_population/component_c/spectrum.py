"""Galaxy spectra: dust, exponential SFR, stellar mass track."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from metcalfe.stella_population.component_c.config import (
    APPLY_DUST_TO_TARGET_COLOUR,
    DUST_REFERENCE_AA,
    GalaxyType,
)
from metcalfe.stella_population.sfr import exponential_sfr_spectrum


def dust_factor(wavelengths_aa, a_b):
    return 10.0 ** (-0.4 * a_b * DUST_REFERENCE_AA / np.asarray(wavelengths_aa, dtype=float))


def galaxy_spectrum(ssp, galaxy_type: GalaxyType, tau_gyr, age_gyr, with_dust=True):
    """Rest-frame spectrum of one type at a given age (per Msun formed in total)."""
    spectrum = exponential_sfr_spectrum(ssp, tau_gyr, age_gyr)
    if with_dust and galaxy_type.has_dust and APPLY_DUST_TO_TARGET_COLOUR:
        spectrum = spectrum * dust_factor(ssp.wavelengths_aa, galaxy_type.a_b)
    return spectrum


def stellar_mass(ssp, tau_gyr, age_gyr):
    """Mass in stars and remnants per Msun formed, from the SSP mass track."""
    mass_track = ssp.extras["stellar_mass"]
    track = replace(
        ssp,
        spectra=np.repeat(mass_track[:, None], 1, axis=1),
        wavelengths_aa=np.array([1.0]),
    )
    return float(exponential_sfr_spectrum(track, tau_gyr, age_gyr)[0])