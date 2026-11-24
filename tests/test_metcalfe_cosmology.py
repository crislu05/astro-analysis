"""Metcalfe Component A validation suite."""

from metcalfe.cosmology.validate import run_validation


def test_metcalfe_cosmology_validation():
    assert run_validation()