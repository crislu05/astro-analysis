"""Non-evolving count integral for faint-limit convergence checks."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from metcalfe.cosmology.config import SR_PER_DEG2
from metcalfe.cosmology.core import Cosmology
from metcalfe.luminosity import config


def non_evolving_counts_per_deg2(
    cosmology: Cosmology,
    luminosity_functions,
    apparent_magnitude,
    m_faint=config.M_FAINT_DEFAULT,
    z_high=15.0,
):
    target_modulus = apparent_magnitude - m_faint
    z_min = brentq(
        lambda z: float(cosmology.distance_modulus(z)) - target_modulus,
        1e-12,
        z_high,
    )

    def integrand_in_ln_z(ln_z):
        z = np.exp(ln_z)
        absolute_magnitude = apparent_magnitude - float(cosmology.distance_modulus(z))
        total_phi = sum(float(lf.phi(absolute_magnitude)) for lf in luminosity_functions)
        return total_phi * float(cosmology.dvdz(z)) * z

    total, _ = quad(
        integrand_in_ln_z, np.log(z_min), np.log(z_high), epsabs=0, epsrel=1e-9, limit=500
    )
    return total * SR_PER_DEG2