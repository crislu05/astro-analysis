"""Component C synthetic self-tests (no BC95 files)."""

from metcalfe.stella_population.component_c.selftest import run_selftests


def test_component_c_selftests():
    assert run_selftests()