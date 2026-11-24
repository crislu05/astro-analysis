"""Synthetic photometry and auxiliary BC95 tables."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Filter:
    name: str
    wavelengths_aa: np.ndarray
    transmission: np.ndarray


def band_flux(wavelengths_aa, spectrum_per_aa, band, redshift=0.0, photon_counting=True):
    rest_wavelengths = band.wavelengths_aa / (1.0 + redshift)
    source = (
        np.interp(rest_wavelengths, wavelengths_aa, spectrum_per_aa, left=0.0, right=0.0)
        / (1.0 + redshift)
    )
    weight = band.transmission * (band.wavelengths_aa if photon_counting else 1.0)
    return np.trapezoid(weight * source, band.wavelengths_aa)


def read_vega_spectrum(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    table = np.array(rows)
    return table[:, 0], table[:, 1]


def vega_magnitude(
    wavelengths_aa,
    spectrum,
    band,
    vega,
    redshift=0.0,
    photon_counting: bool = False,
) -> float:
    """Apparent Vega magnitude for a source spectrum (BC95 tables use energy counting)."""
    vega_wavelengths, vega_flux = vega
    source_flux = band_flux(wavelengths_aa, spectrum, band, redshift, photon_counting)
    vega_band_flux = band_flux(vega_wavelengths, vega_flux, band, 0.0, photon_counting)
    return -2.5 * np.log10(source_flux / vega_band_flux)


def vega_colour(wavelengths_aa, spectrum, band_one, band_two, vega, redshift=0.0, photon_counting=True):
    vega_wavelengths, vega_flux = vega
    source_ratio = band_flux(wavelengths_aa, spectrum, band_one, redshift, photon_counting) / band_flux(
        wavelengths_aa, spectrum, band_two, redshift, photon_counting
    )
    vega_ratio = band_flux(vega_wavelengths, vega_flux, band_one, 0.0, photon_counting) / band_flux(
        vega_wavelengths, vega_flux, band_two, 0.0, photon_counting
    )
    return -2.5 * np.log10(source_ratio / vega_ratio)


def ab_colour(wavelengths_aa, spectrum, band_one, band_two, photon_counting=True):
    flat_fnu = 1.0 / wavelengths_aa**2
    source_ratio = band_flux(wavelengths_aa, spectrum, band_one, 0.0, photon_counting) / band_flux(
        wavelengths_aa, spectrum, band_two, 0.0, photon_counting
    )
    reference_ratio = band_flux(wavelengths_aa, flat_fnu, band_one, 0.0, photon_counting) / band_flux(
        wavelengths_aa, flat_fnu, band_two, 0.0, photon_counting
    )
    return -2.5 * np.log10(source_ratio / reference_ratio)


def top_hat(name, centre_aa, width_aa, points=200):
    wavelengths = np.linspace(centre_aa - width_aa / 2, centre_aa + width_aa / 2, points)
    return Filter(name, wavelengths, np.ones(points))


def read_3color_table(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 11:
            try:
                rows.append([float(value) for value in parts[:7]])
            except ValueError:
                continue
    return np.array(rows)


def read_1color_table(path):
    """SSP rest-frame U, B, V, K Vega magnitudes from bc95_ssp_sp_1.1color."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            log_age = float(parts[0])
        except ValueError:
            continue
        if log_age < 4.0:
            continue
        try:
            mbol, umag, bmag, vmag, kmag = (float(parts[i]) for i in range(1, 6))
        except ValueError:
            continue
        rows.append((log_age, mbol, umag, bmag, vmag, kmag))
    return np.array(rows, dtype=float)


def fit_tau_to_colour(
    spectrum_at_tau,
    colour_at_spectrum,
    target_colour: float,
    tau_min_gyr: float = 0.05,
    tau_max_gyr: float = 30.0,
    n_grid: int = 800,
) -> float:
    """Grid search for tau (Gyr) that reproduces target_colour."""
    taus = np.geomspace(tau_min_gyr, tau_max_gyr, n_grid)
    errors = [abs(colour_at_spectrum(spectrum_at_tau(tau)) - target_colour) for tau in taus]
    return float(taus[int(np.argmin(errors))])


def k_correction(
    wavelengths_aa,
    spectrum,
    band,
    vega,
    redshift: float,
    photon_counting: bool = False,
) -> float:
    """K = m(z) - m(0) for the same rest-frame spectrum (BC95 energy-counting default)."""
    if redshift == 0.0:
        return 0.0
    magnitude_at_z = vega_magnitude(wavelengths_aa, spectrum, band, vega, redshift, photon_counting)
    magnitude_at_zero = vega_magnitude(wavelengths_aa, spectrum, band, vega, 0.0, photon_counting)
    return magnitude_at_z - magnitude_at_zero


def powerlaw_spectrum_integral(wavelengths, spectrum):
    total = 0.0
    for start, end, flux_start, flux_end in zip(
        wavelengths[:-1], wavelengths[1:], spectrum[:-1], spectrum[1:]
    ):
        if flux_start <= 0 or flux_end <= 0:
            total += 0.5 * (flux_start + flux_end) * (end - start)
            continue
        slope = np.log(flux_end / flux_start) / np.log(end / start)
        if abs(slope + 1) < 1e-8:
            total += flux_start * start * np.log(end / start)
        else:
            total += flux_start * start / (slope + 1) * ((end / start) ** (slope + 1) - 1)
    return total