"""Weighted least-squares fit for global LF normalisation A."""

import numpy as np


def fit_normalisation(model_counts, data_counts, data_errors):
    model_counts = np.asarray(model_counts, dtype=float)
    data_counts = np.asarray(data_counts, dtype=float)
    weights = 1.0 / np.asarray(data_errors, dtype=float) ** 2

    curvature = np.sum(weights * model_counts**2)
    best_factor = np.sum(weights * data_counts * model_counts) / curvature
    factor_error = 1.0 / np.sqrt(curvature)

    residuals = data_counts - best_factor * model_counts
    chi2 = float(np.sum(weights * residuals**2))
    degrees_of_freedom = len(data_counts) - 1
    return best_factor, factor_error, chi2, degrees_of_freedom