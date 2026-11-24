"""Schechter luminosity function (Metcalfe et al. 2001, rest b-band)."""

from dataclasses import dataclass, replace

import numpy as np

from metcalfe.luminosity import config
from metcalfe.luminosity.gamma import upper_incomplete_gamma

LN10 = np.log(10.0)


@dataclass(frozen=True)
class SchechterLF:
    name: str
    phi_star: float
    m_star: float
    alpha: float
    h: float

    def x(self, absolute_magnitude):
        return 10.0 ** (0.4 * (self.m_star - np.asarray(absolute_magnitude, dtype=float)))

    def phi(self, absolute_magnitude):
        x = self.x(absolute_magnitude)
        return 0.4 * LN10 * self.phi_star * x ** (self.alpha + 1.0) * np.exp(-x)

    def number_density(
        self,
        m_bright=config.M_BRIGHT_DEFAULT,
        m_faint=config.M_FAINT_DEFAULT,
    ):
        x_faint = float(self.x(m_faint))
        x_bright = float(self.x(m_bright))
        a = self.alpha + 1.0
        return self.phi_star * (
            upper_incomplete_gamma(a, x_faint) - upper_incomplete_gamma(a, x_bright)
        )

    def luminosity_density_in_l_star(
        self,
        m_bright=config.M_BRIGHT_DEFAULT,
        m_faint=config.M_FAINT_DEFAULT,
    ):
        x_faint = float(self.x(m_faint))
        x_bright = float(self.x(m_bright))
        a = self.alpha + 2.0
        return self.phi_star * (
            upper_incomplete_gamma(a, x_faint) - upper_incomplete_gamma(a, x_bright)
        )

    def rescale_h(self, h_new):
        ratio = h_new / self.h
        return replace(
            self,
            phi_star=self.phi_star * ratio**3,
            m_star=self.m_star + 5.0 * np.log10(ratio),
            h=h_new,
        )

    def scale_normalisation(self, factor):
        return replace(self, phi_star=self.phi_star * factor)

    def sample(
        self,
        number,
        rng,
        m_bright=config.M_BRIGHT_DEFAULT,
        m_faint=config.M_FAINT_DEFAULT,
        grid_step=1e-3,
    ):
        grid = np.arange(m_bright, m_faint + grid_step, grid_step)
        density = self.phi(grid)
        cumulative = np.concatenate(
            [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))]
        )
        cumulative /= cumulative[-1]
        return np.interp(rng.random(number), cumulative, grid)