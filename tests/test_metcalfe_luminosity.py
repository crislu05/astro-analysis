"""Metcalfe Component B validation."""

from metcalfe.luminosity.validate import run_validation


def test_metcalfe_luminosity_validation():
    assert run_validation()