"""Exponential star formation history convolved with SSP spectra."""

import numpy as np

from metcalfe.stella_population import config
from metcalfe.stella_population.ised import StellarPopulation


def _segment_weights(interval_over_tau):
    x = np.asarray(interval_over_tau, dtype=float)
    one_minus_exp = -np.expm1(-x)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_exact = (one_minus_exp - x * np.exp(-x)) / x
    g_series = x / 2.0 - x**2 / 3.0 + x**3 / 8.0
    g = np.where(x < 1e-4, g_series, g_exact)
    return one_minus_exp - g, g


def exponential_sfr_spectrum(ssp: StellarPopulation, tau_gyr, age_gyr):
    if ssp.sfr_code != 0:
        raise ValueError(f"{ssp.name} is not a single burst")
    if not 0.0 <= age_gyr <= ssp.ages_gyr[-1]:
        raise ValueError(f"age must lie in [0, {ssp.ages_gyr[-1]:.2f}] Gyr")
    if age_gyr == 0.0:
        return np.zeros_like(ssp.wavelengths_aa)

    tau_yr = tau_gyr * config.YEARS_PER_GYR
    age_yr = age_gyr * config.YEARS_PER_GYR

    inside = ssp.ages_yr < age_yr
    node_ages = np.append(ssp.ages_yr[inside], age_yr)
    last_index = np.count_nonzero(inside) - 1
    if age_yr == ssp.ages_yr[last_index + 1 if last_index + 1 < len(ssp.ages_yr) else last_index]:
        end_spectrum = ssp.spectra[last_index + 1]
    else:
        lower, upper = ssp.ages_yr[last_index], ssp.ages_yr[last_index + 1]
        fraction = (age_yr - lower) / (upper - lower)
        end_spectrum = (1.0 - fraction) * ssp.spectra[last_index] + fraction * ssp.spectra[last_index + 1]
    node_spectra = np.vstack([ssp.spectra[inside], end_spectrum])

    start_ages = node_ages[:-1]
    end_ages = node_ages[1:]
    interval_over_tau = (end_ages - start_ages) / tau_yr
    decay_at_end = np.exp(-(age_yr - end_ages) / tau_yr)
    weight_end, weight_start = _segment_weights(interval_over_tau)

    weights = np.zeros(len(node_ages))
    weights[1:] += decay_at_end * weight_end
    weights[:-1] += decay_at_end * weight_start
    return weights @ node_spectra


def exponential_sfr_mass_formed(tau_gyr, age_gyr):
    return -np.expm1(-age_gyr / tau_gyr)


def bc95_trapezoid_convolution(ssp: StellarPopulation, tau_gyr):
    tau_yr = tau_gyr * config.YEARS_PER_GYR
    ages = ssp.ages_yr
    output = np.zeros_like(ssp.spectra)
    for current in range(1, len(ages)):
        weights = np.zeros(current + 1)
        for k in range(current + 1):
            if k == 0:
                interval = ages[1] - ages[0]
            elif k == current:
                interval = ages[current] - ages[current - 1]
            else:
                interval = ages[k + 1] - ages[k - 1]
            star_formation_rate = np.exp(-(ages[current] - ages[k]) / tau_yr) / tau_yr
            weights[k] = star_formation_rate * interval / 2.0
        output[current] = weights @ ssp.spectra[: current + 1]
    return output