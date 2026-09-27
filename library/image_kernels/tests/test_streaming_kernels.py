"""Regression coverage for rolling stereo costs and denoising patch sums."""
import numpy as np
import pytest
from viame import image_kernels as kernels


@pytest.mark.parametrize('mode', ['sgbm', 'hh', 'sgbm_3way'])
@pytest.mark.parametrize('channels', [1, 3])
def test_stereo_streaming_preserves_opencv_result(mode, channels):
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(81)
    shape = (65, 96) if channels == 1 else (65, 96, channels)
    left = rng.integers(0, 256, shape, dtype=np.uint8)
    right = np.roll(left, -4, axis=1)
    reference = cv2.StereoSGBM_create(
        minDisparity=-3, numDisparities=16, blockSize=5,
        P1=8 * channels * 25, P2=32 * channels * 25,
        mode={'sgbm': cv2.STEREO_SGBM_MODE_SGBM,
              'hh': cv2.STEREO_SGBM_MODE_HH,
              'sgbm_3way': cv2.STEREO_SGBM_MODE_SGBM_3WAY}[mode]).compute(left, right)
    found = kernels.stereo_sgbm(left, right, min_disparity=-3,
                               num_disparities=16, block_size=5,
                               p1=8 * channels * 25, p2=32 * channels * 25, mode=mode)
    assert np.array_equal(found, reference)


@pytest.mark.parametrize('channels', [1, 2, 3])
@pytest.mark.parametrize('patch,window', [(1, 3), (3, 7), (7, 21)])
def test_denoise_rolling_sums_match_opencv(channels, patch, window):
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(93)
    shape = (19, 27) if channels == 1 else (19, 27, channels)
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    found = kernels.denoise(image, 12, patch, window)
    reference = cv2.fastNlMeansDenoising(image, None, 12, patch, window)
    assert np.array_equal(found, reference)


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


@pytest.mark.parametrize('radius', [0, 2, 5, 10])
@pytest.mark.parametrize('channels', [1, 3])
def test_wls_rolling_confidence_matches_opencv(radius, channels):
    cv2 = pytest.importorskip('cv2')
    if not hasattr(cv2, 'ximgproc'):
        pytest.skip('OpenCV contrib is required for the WLS reference')
    rng = np.random.default_rng(42)
    shape = (17, 47) if channels == 1 else (17, 47, channels)
    guide = rng.integers(0, 32, shape, dtype=np.uint8)
    left = rng.integers(240, 273, (17, 47), dtype=np.int16)
    right = np.full(left.shape, -256, np.int16)
    reference = cv2.ximgproc.createDisparityWLSFilterGeneric(True)
    reference.setDepthDiscontinuityRadius(radius)
    reference.setLambda(8000)
    reference.setSigmaColor(1)
    expected = reference.filter(left, guide, disparity_map_right=right,
                                ROI=(16, 0, 31, 17))
    actual = kernels.filter_disparity_wls(
        guide, left, right, left_offset=16, discontinuity_radius=radius)
    # OpenCV rounds its result back into int16 disparity units.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.501)
