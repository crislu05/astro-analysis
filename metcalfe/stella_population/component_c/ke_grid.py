"""C6: [K + E] and evolution grids vs redshift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metcalfe.cosmology.core import Cosmology
from metcalfe.stella_population.component_c.config import (
    AGE_FLOOR_GYR,
    GALAXY_TYPES,
    KE_DZ,
    PHOTON_COUNTING,
)
from metcalfe.stella_population.component_c.spectrum import galaxy_spectrum
from metcalfe.stella_population.photometry import Filter, band_flux


@dataclass
class KEGrid:
    config_name: str
    redshifts: np.ndarray
    band_names: list
    type_names: list
    k_plus_e: np.ndarray
    k_only: np.ndarray
    evolution: np.ndarray


def k_plus_e_grid(
    ssp,
    cosmology: Cosmology,
    t_gal_gyr,
    taus,
    band_list,
    galaxy_types=GALAXY_TYPES,
    dz=KE_DZ,
    with_dust=True,
):
    """taus: dict type name -> tau [Gyr]; band_list: list of (name, Filter)."""
    z_formation = cosmology.z_formation(t_gal_gyr)
    redshifts = np.arange(0.0, z_formation + 0.5 * dz, dz)
    redshifts = redshifts[redshifts <= z_formation]
    ages = t_gal_gyr - np.array([float(cosmology.lookback_time(z)) for z in redshifts])
    shape = (len(galaxy_types), len(band_list), len(redshifts))
    k_plus_e = np.full(shape, np.inf)
    k_only = np.full(shape, np.nan)
    wavelengths = ssp.wavelengths_aa

    for type_index, galaxy_type in enumerate(galaxy_types):
        tau = taus[galaxy_type.name]
        today = galaxy_spectrum(ssp, galaxy_type, tau, t_gal_gyr, with_dust)
        reference = [
            band_flux(wavelengths, today, band, 0.0, PHOTON_COUNTING) for _, band in band_list
        ]
        for z_index, (z, age) in enumerate(zip(redshifts, ages)):
            if age < AGE_FLOOR_GYR:
                continue
            young = galaxy_spectrum(ssp, galaxy_type, tau, age, with_dust)
            for band_index, (_, band) in enumerate(band_list):
                flux_young = band_flux(wavelengths, young, band, z, PHOTON_COUNTING)
                flux_today_shifted = band_flux(wavelengths, today, band, z, PHOTON_COUNTING)
                k_plus_e[type_index, band_index, z_index] = -2.5 * np.log10(
                    flux_young / reference[band_index]
                )
                k_only[type_index, band_index, z_index] = -2.5 * np.log10(
                    flux_today_shifted / reference[band_index]
                )

    return KEGrid(
        config_name="",
        redshifts=redshifts,
        band_names=[name for name, _ in band_list],
        type_names=[galaxy_type.name for galaxy_type in galaxy_types],
        k_plus_e=k_plus_e,
        k_only=k_only,
        evolution=k_plus_e - k_only,
    )