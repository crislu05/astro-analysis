"""Reference formulas for validation (Mattig, Hogg, toy counts)."""

import numpy as np
from scipy.integrate import quad

from metcalfe.cosmology.config import SPEED_OF_LIGHT_KM_S
from metcalfe.cosmology.core import Cosmology


def mattig_luminosity_distance(z, q0, h0):
    z = np.asarray(z, dtype=float)
    hubble_distance = SPEED_OF_LIGHT_KM_S / h0
    return hubble_distance / q0**2 * (
        q0 * z + (q0 - 1.0) * (np.sqrt(1.0 + 2.0 * q0 * z) - 1.0)
    )


def open_universe_comoving_volume(cosmology: Cosmology, z):
    hubble_distance = cosmology.hubble_distance_mpc
    sqrt_omega_k = np.sqrt(cosmology.omega_k)
    transverse_over_dh = cosmology.transverse_comoving_distance(z) / hubble_distance
    prefactor = 4.0 * np.pi * hubble_distance**3 / (2.0 * cosmology.omega_k)
    term = transverse_over_dh * np.sqrt(1.0 + cosmology.omega_k * transverse_over_dh**2)
    term -= np.arcsinh(sqrt_omega_k * transverse_over_dh) / sqrt_omega_k
    return prefactor * term


def schechter_per_magnitude(absolute_magnitude, phi_star, m_star, alpha):
    x = 10.0 ** (0.4 * (m_star - absolute_magnitude))
    return 0.4 * np.log(10.0) * phi_star * x ** (alpha + 1.0) * np.exp(-x)


def non_evolving_counts(
    cosmology: Cosmology,
    apparent_magnitude,
    phi_star,
    m_star,
    alpha,
    z_low=1e-9,
    z_high=6.0,
):
    def integrand_in_ln_z(ln_z):
        z = np.exp(ln_z)
        absolute_magnitude = apparent_magnitude - float(cosmology.distance_modulus(z))
        return (
            schechter_per_magnitude(absolute_magnitude, phi_star, m_star, alpha)
            * float(cosmology.dvdz(z))
            * z
        )

    total, _ = quad(
        integrand_in_ln_z, np.log(z_low), np.log(z_high), epsabs=0, epsrel=1e-9, limit=500
    )
    return total