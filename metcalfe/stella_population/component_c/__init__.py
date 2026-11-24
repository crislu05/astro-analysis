"""Metcalfe Component C §C2–C6: tau fits, legacy regression, [K+E] grids."""

from metcalfe.stella_population.component_c.run import run_on_bc95
from metcalfe.stella_population.component_c.selftest import run_selftests

__all__ = ["run_selftests", "run_on_bc95"]