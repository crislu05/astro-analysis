"""Wrap photometry synthetic self-tests for pytest."""

from pipeline.photometry.selftest import selftest


def test_photometry_selftest():
    selftest()