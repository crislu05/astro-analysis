"""Synthetic-data tests for pipeline.detect (lab script section 6.1)."""

import numpy as np
import pytest

from pipeline.detect import detect
from pipeline.detect.synthetic import add_gaussian, match, noise_image

SHAPE = (600, 600)
SKY, SIGMA, FWHM = 3420.0, 10.0, 4.0
NOMASK = np.zeros(SHAPE, dtype=bool)
MAX_NOISE_EXTRA = 2


def matched(cat, x, y, radius=2.0):
    return match(
        np.atleast_1d(x),
        np.atleast_1d(y),
        np.asarray(cat["x"]),
        np.asarray(cat["y"]),
        radius,
    )


def n_found(result):
    return len(result.catalogue)


def test_all_zeros_finds_nothing():
    assert n_found(detect(np.zeros(SHAPE), NOMASK)) == 0


def test_constant_image_finds_nothing():
    assert n_found(detect(np.full(SHAPE, SKY), NOMASK)) == 0


def test_pure_noise_false_rate_below_4_per_mpix():
    n_img = 6
    total = sum(
        n_found(detect(noise_image(SHAPE, SKY, SIGMA, seed=s), NOMASK))
        for s in range(n_img)
    )
    rate = total / (n_img * SHAPE[0] * SHAPE[1] / 1e6)
    assert rate < 4.0, f"{rate:.1f} false detections per Mpix"


def test_single_source_position_and_count():
    img = noise_image(SHAPE, SKY, SIGMA, seed=1)
    add_gaussian(img, 300.3, 250.7, flux=20000, fwhm=FWHM)
    cat = detect(img, NOMASK).catalogue
    j = matched(cat, 300.3, 250.7)[0]
    assert j >= 0 and len(cat) <= 1 + MAX_NOISE_EXTRA
    assert abs(cat["x"][j] - 300.3) < 0.2
    assert abs(cat["y"][j] - 250.7) < 0.2
    assert 0.85 < cat["flux"][j] / 20000 < 1.05


def make_grid(fluxes, seed=2):
    img = noise_image(SHAPE, SKY, SIGMA, seed=seed)
    xs, ys, fs = [], [], []
    rng = np.random.default_rng(seed)
    k = 0
    for gy in range(50, 600, 50):
        for gx in range(50, 600, 50):
            f = fluxes[k % len(fluxes)]
            x, y = gx + rng.uniform(-0.5, 0.5), gy + rng.uniform(-0.5, 0.5)
            add_gaussian(img, x, y, f, FWHM)
            xs.append(x)
            ys.append(y)
            fs.append(f)
            k += 1
    return img, np.array(xs), np.array(ys), np.array(fs)


def test_bright_sources_all_recovered_and_no_spurious():
    img, xs, ys, fs = make_grid([5000, 10000, 50000])
    cat = detect(img, NOMASK).catalogue
    idx = match(xs, ys, np.asarray(cat["x"]), np.asarray(cat["y"]), 2.0)
    assert (idx >= 0).all(), f"missed {np.sum(idx < 0)} bright sources"
    assert len(set(idx)) == len(xs), "a bright source was split or merged"
    assert len(cat) - len(xs) <= MAX_NOISE_EXTRA


def test_completeness_falls_with_flux():
    img, xs, ys, fs = make_grid([150, 300, 600, 1200])
    cat = detect(img, NOMASK).catalogue
    idx = match(xs, ys, np.asarray(cat["x"]), np.asarray(cat["y"]), 2.0)
    frac = {f: np.mean(idx[fs == f] >= 0) for f in [150, 300, 600, 1200]}
    assert frac[1200] == 1.0
    assert frac[150] < frac[600] <= frac[1200]


def test_close_pair_is_deblended():
    img = noise_image(SHAPE, SKY, SIGMA, seed=3)
    add_gaussian(img, 300, 300, 30000, FWHM)
    add_gaussian(img, 309, 300, 15000, FWHM)
    cat = detect(img, NOMASK).catalogue
    idx = matched(cat, [300, 309], [300, 300], radius=1.5)
    assert (idx >= 0).all() and idx[0] != idx[1], "pair not separated"


def test_masked_sources_are_ignored_and_do_not_leak():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[250:351, 250:351] = True
    cats = []
    for with_blob in (False, True):
        img = noise_image(SHAPE, SKY, SIGMA, seed=4)
        add_gaussian(img, 100, 100, 20000, FWHM)
        if with_blob:
            add_gaussian(img, 300, 300, 5e6, 20.0)
        cats.append(detect(img, mask).catalogue)
    assert matched(cats[1], 100, 100)[0] >= 0
    assert len(cats[0]) == len(cats[1])
    assert np.allclose(np.sort(cats[0]["x"]), np.sort(cats[1]["x"]), atol=0.01)


def test_near_mask_flag():
    img = noise_image(SHAPE, SKY, SIGMA, seed=5)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:, 290:300] = True
    add_gaussian(img, 304, 200, 20000, FWHM)
    add_gaussian(img, 450, 200, 20000, FWHM)
    cat = detect(img, mask).catalogue
    j_near, j_far = matched(cat, [304, 450], [200, 200])
    assert cat["near_mask"][j_near] and not cat["near_mask"][j_far]


def test_negative_image_finds_nothing():
    img = noise_image(SHAPE, SKY, SIGMA, seed=6)
    add_gaussian(img, 300, 300, 20000, FWHM)
    flipped = 2 * SKY - img
    cat = detect(flipped, NOMASK).catalogue
    assert len(cat) <= MAX_NOISE_EXTRA
    d = np.hypot(np.asarray(cat["x"]) - 300, np.asarray(cat["y"]) - 300)
    assert (d > 10).all()