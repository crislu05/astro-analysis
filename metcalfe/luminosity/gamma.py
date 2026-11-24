"""Upper incomplete gamma for Schechter integrals (alpha <= 0 included)."""

import numpy as np
from scipy.special import exp1, gamma, gammaincc


def upper_incomplete_gamma(a, x):
    """Gamma(a, x) = int_x^inf t^(a-1) e^-t dt."""
    if x <= 0.0:
        raise ValueError("x must be positive")
    if a > 0.0:
        return gamma(a) * gammaincc(a, x)
    if a == 0.0:
        return exp1(x)
    return (upper_incomplete_gamma(a + 1.0, x) - x**a * np.exp(-x)) / a