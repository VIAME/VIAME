"""Regression coverage for rolling stereo costs and denoising patch sums.

The checks against an outside reference implementation are in
`tests/reference/test_kernels_streaming.py`.
"""
import numpy as np
import pytest
from viame import image_kernels as kernels


@pytest.mark.parametrize('minimum', [0, -3])
def test_wls_zero_confidence_returns_finite_invalid_disparity(minimum):
    guide = np.zeros((12, 40), np.uint8)
    left = np.full(guide.shape, 256, np.int16)
    right = np.zeros(guide.shape, np.int16)
    result = kernels.filter_disparity_wls(
        guide, left, right, left_offset=16, min_disparity=minimum,
        discontinuity_radius=2)
    assert np.isfinite(result).all()
    assert np.all(result == (minimum - 1) * 16)


def test_wls_consistent_disparities_keep_their_value():
    guide = np.zeros((12, 40), np.uint8)
    left = np.full(guide.shape, 256, np.int16)
    right = np.full(guide.shape, -256, np.int16)
    result = kernels.filter_disparity_wls(
        guide, left, right, left_offset=16, discontinuity_radius=2)
    assert np.all(result[:, :16] == -16)
    np.testing.assert_allclose(result[:, 16:], 256, rtol=1e-6)


@pytest.mark.parametrize('channels', [1, 2, 3])
def test_zero_strength_denoising_preserves_image(channels):
    rng = np.random.default_rng(4)
    shape = (24, 32) if channels == 1 else (24, 32, channels)
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    assert np.array_equal(kernels.denoise(image, 0, 7, 21), image)


@pytest.mark.parametrize('parameter,value', [
    ('lambda_', -1), ('lambda_', np.nan), ('lambda_', np.inf),
    ('sigma', -1), ('sigma', 0), ('sigma', np.nan), ('sigma', np.inf)])
@pytest.mark.parametrize('wls', [False, True])
def test_smoother_rejects_invalid_parameters(parameter, value, wls):
    guide = np.zeros((8, 8), np.uint8)
    kwargs = {'lambda_': 8000, 'sigma': 1, parameter: value}
    with pytest.raises(ValueError, match='lambda.*sigma'):
        if wls:
            disparity = np.zeros(guide.shape, np.int16)
            kernels.filter_disparity_wls(guide, disparity, disparity, **kwargs)
        else:
            kernels.smooth_globally(guide, np.ones(guide.shape, np.float32), **kwargs)


def test_smoother_zero_lambda_is_identity():
    rng = np.random.default_rng(87)
    guide = rng.integers(0, 256, (9, 11), dtype=np.uint8)
    image = rng.random((9, 11, 2), dtype=np.float32)
    assert np.array_equal(kernels.smooth_globally(guide, image, 0, 1), image)
