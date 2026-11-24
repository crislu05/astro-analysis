"""BC95 validation (requires data files)."""

import pytest

from metcalfe.stella_population.data_paths import default_bc95_paths
from metcalfe.stella_population.validate import run_validation


@pytest.mark.skipif(
    bool(default_bc95_paths().missing()),
    reason="BC95 .ised files not in data/bc95 (see --show-data-help)",
)
def test_metcalfe_stella_population_validation():
    assert run_validation(default_bc95_paths())