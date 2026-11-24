"""Cosmology presets: legacy (Metcalfe 2001) and updated (Planck 2018)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

SPEED_OF_LIGHT_KM_S = 299_792.458
HUBBLE_TIME_GYR_FOR_H0_EQUAL_1 = 977.792_221_680_789_1
SR_PER_DEG2 = (np.pi / 180.0) ** 2
MPC_IN_PC = 1.0e6

DEFAULT_Z_MAX = 15.0
DEFAULT_DZ = 0.005


@dataclass(frozen=True)
class CosmologyParams:
    name: str
    omega_m: float
    omega_lambda: float
    h0: float
    t_gal_gyr: Optional[float]

    @property
    def omega_k(self) -> float:
        return 1.0 - self.omega_m - self.omega_lambda

    @property
    def q0(self) -> float:
        return 0.5 * self.omega_m - self.omega_lambda

    @property
    def h(self) -> float:
        return self.h0 / 100.0


LEGACY_Q005 = CosmologyParams("Legacy q0=0.05", 0.1, 0.0, 50.0, 16.0)
LEGACY_LAMBDA = CosmologyParams("Legacy Lambda", 0.3, 0.7, 50.0, 18.0)
LEGACY_Q05 = CosmologyParams("Legacy q0=0.5", 1.0, 0.0, 50.0, None)
UPDATED = CosmologyParams("Updated Planck 2018", 0.315, 0.685, 67.4, 13.0)
UPDATED_TGAL_12P5 = CosmologyParams("Updated, t_gal=12.5", 0.315, 0.685, 67.4, 12.5)

ALL_CONFIGS = [LEGACY_Q005, LEGACY_LAMBDA, LEGACY_Q05, UPDATED, UPDATED_TGAL_12P5]