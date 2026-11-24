"""Tabulated Friedmann cosmology (Metcalfe et al. 2001, Component A)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from metcalfe.cosmology import config
from metcalfe.cosmology.config import CosmologyParams, MPC_IN_PC, SR_PER_DEG2


class Cosmology:
    def __init__(
        self,
        params: CosmologyParams,
        z_max: float = config.DEFAULT_Z_MAX,
        dz: float = config.DEFAULT_DZ,
    ):
        self.params = params
        self.omega_m = params.omega_m
        self.omega_lambda = params.omega_lambda
        self.omega_k = params.omega_k
        self.h0 = params.h0

        self.hubble_distance_mpc = config.SPEED_OF_LIGHT_KM_S / self.h0
        self.hubble_time_gyr = config.HUBBLE_TIME_GYR_FOR_H0_EQUAL_1 / self.h0

        number_of_steps = int(round(z_max / dz))
        self.z_grid = np.linspace(0.0, z_max, number_of_steps + 1)
        self.z_max = self.z_grid[-1]

        comoving_table, lookback_table = self._build_tables()
        self.comoving_distance_table = comoving_table
        self.lookback_time_table = lookback_table

        end_points = np.array([0.0, self.z_max])
        comoving_end_slopes = self.hubble_distance_mpc * self._inverse_E(end_points)
        lookback_end_slopes = self.hubble_time_gyr * self._lookback_integrand(end_points)
        self._comoving_spline = CubicSpline(
            self.z_grid,
            comoving_table,
            bc_type=((1, comoving_end_slopes[0]), (1, comoving_end_slopes[1])),
        )
        self._lookback_spline = CubicSpline(
            self.z_grid,
            lookback_table,
            bc_type=((1, lookback_end_slopes[0]), (1, lookback_end_slopes[1])),
        )

    def E(self, z):
        z = np.asarray(z, dtype=float)
        one_plus_z = 1.0 + z
        return np.sqrt(
            self.omega_m * one_plus_z**3
            + self.omega_k * one_plus_z**2
            + self.omega_lambda
        )

    def _inverse_E(self, z):
        return 1.0 / self.E(z)

    def _lookback_integrand(self, z):
        return 1.0 / ((1.0 + z) * self.E(z))

    def _build_tables(self):
        number_of_points = len(self.z_grid)
        comoving_dimensionless = np.zeros(number_of_points)
        lookback_dimensionless = np.zeros(number_of_points)

        for index in range(1, number_of_points):
            z_low = self.z_grid[index - 1]
            z_high = self.z_grid[index]
            comoving_step, _ = quad(
                self._inverse_E, z_low, z_high, epsabs=0, epsrel=1e-12
            )
            lookback_step, _ = quad(
                self._lookback_integrand, z_low, z_high, epsabs=0, epsrel=1e-12
            )
            comoving_dimensionless[index] = comoving_dimensionless[index - 1] + comoving_step
            lookback_dimensionless[index] = lookback_dimensionless[index - 1] + lookback_step

        return (
            self.hubble_distance_mpc * comoving_dimensionless,
            self.hubble_time_gyr * lookback_dimensionless,
        )

    def _check_range(self, z):
        z = np.asarray(z, dtype=float)
        if np.any(z < 0.0) or np.any(z > self.z_max):
            raise ValueError(f"z must lie in [0, {self.z_max}]")
        return z

    def comoving_distance(self, z):
        z = self._check_range(z)
        return self._comoving_spline(z)

    def transverse_comoving_distance(self, z):
        comoving = self.comoving_distance(z)
        if abs(self.omega_k) < 1e-12:
            return comoving
        sqrt_abs_omega_k = np.sqrt(abs(self.omega_k))
        argument = sqrt_abs_omega_k * comoving / self.hubble_distance_mpc
        if self.omega_k > 0:
            return self.hubble_distance_mpc / sqrt_abs_omega_k * np.sinh(argument)
        return self.hubble_distance_mpc / sqrt_abs_omega_k * np.sin(argument)

    def luminosity_distance(self, z):
        z = self._check_range(z)
        return (1.0 + z) * self.transverse_comoving_distance(z)

    def distance_modulus(self, z):
        luminosity_distance_pc = self.luminosity_distance(z) * MPC_IN_PC
        with np.errstate(divide="ignore"):
            return 5.0 * np.log10(luminosity_distance_pc / 10.0)

    def dvdz(self, z):
        z = self._check_range(z)
        transverse = self.transverse_comoving_distance(z)
        return self.hubble_distance_mpc * transverse**2 / self.E(z)

    def dvdz_per_deg2(self, z):
        return self.dvdz(z) * SR_PER_DEG2

    def comoving_volume_all_sky(self, z_high):
        volume_per_sr, _ = quad(
            lambda z: float(self.dvdz(z)),
            0.0,
            z_high,
            epsabs=0,
            epsrel=1e-10,
            limit=200,
        )
        return 4.0 * np.pi * volume_per_sr

    def lookback_time(self, z):
        z = self._check_range(z)
        return self._lookback_spline(z)

    def age_of_universe(self):
        integral, _ = quad(
            self._lookback_integrand, 0.0, np.inf, epsabs=0, epsrel=1e-12, limit=200
        )
        return self.hubble_time_gyr * integral

    def z_formation(self, t_gal_gyr: Optional[float] = None):
        if t_gal_gyr is None:
            t_gal_gyr = self.params.t_gal_gyr
        if t_gal_gyr is None:
            raise ValueError(f"{self.params.name} has no galaxy age")
        age = self.age_of_universe()
        if t_gal_gyr >= age:
            raise ValueError(
                f"t_gal = {t_gal_gyr} Gyr is not below the age of the universe ({age:.3f} Gyr)"
            )
        oldest_tabulated = float(self.lookback_time(self.z_max))
        if t_gal_gyr > oldest_tabulated:
            raise ValueError(
                f"t_gal = {t_gal_gyr} Gyr needs z_f > z_max = {self.z_max}; extend the grid"
            )
        return brentq(
            lambda z: float(self.lookback_time(z)) - t_gal_gyr,
            0.0,
            self.z_max,
            xtol=1e-10,
        )