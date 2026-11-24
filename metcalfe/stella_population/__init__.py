"""Component C: stellar populations (BC95) and evolution."""

from metcalfe.stella_population.ised import StellarPopulation, read_ised
from metcalfe.stella_population.sfr import exponential_sfr_spectrum
from metcalfe.stella_population.validate import run_validation

__all__ = ["StellarPopulation", "exponential_sfr_spectrum", "read_ised", "run_validation"]