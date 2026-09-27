"""Numerical and concurrency coverage for kernel fast paths and workspaces."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from viame import image_kernels as kernels


@pytest.mark.parametrize('shape', [(1, 1), (3, 5), (33, 67), (129, 130, 3)])
@pytest.mark.parametrize('size', [1, 3, 5, 9, 21])
@pytest.mark.parametrize('border', ['reflect_101', 'reflect', 'replicate', 'wrap'])
def test_gaussian_workspace_changes_shape(shape, size, border):
    rng = np.random.default_rng(45)
    workspace = kernels.GaussianWorkspace()
    previous = None
    for current in [(40, 80), shape, (3, 2)]:
        image = rng.random(current, dtype=np.float32)
        actual = kernels.gaussian_blur(image, size, 2.5, border, workspace=workspace)
        expected = kernels.gaussian_blur(image, size, 2.5, border)
        assert np.array_equal(actual, expected)
        if previous is not None:
            assert np.array_equal(previous[0], previous[1])
        previous = actual, actual.copy()


@pytest.mark.parametrize('mode', ['sgbm', 'hh', 'sgbm_3way'])
def test_stereo_workspace_resets_costs_and_dimensions(mode):
    rng = np.random.default_rng(28)
    workspace = kernels.StereoWorkspace()
    for shape, count, block, minimum in [((65, 130), 32, 5, -3),
                                         ((12, 43), 16, 1, 0),
                                         ((65, 130), 32, 5, -3)]:
        left = rng.integers(0, 256, shape, dtype=np.uint8)
        right = np.roll(left, -4, axis=1)
        args = dict(mode=mode, num_disparities=count, block_size=block,
                    min_disparity=minimum)
        assert np.array_equal(kernels.stereo_sgbm(left, right, workspace=workspace, **args),
                              kernels.stereo_sgbm(left, right, **args))


def test_concurrent_kernels_and_shared_workspaces():
    rng = np.random.default_rng(91)
    image = rng.random((151, 179), dtype=np.float32)
    left = rng.integers(0, 256, (65, 97), dtype=np.uint8)
    right = np.roll(left, -4, axis=1)
    gaussian = kernels.GaussianWorkspace()
    stereo = kernels.StereoWorkspace()
    expected_g = kernels.gaussian_blur(image, 9, 2)
    expected_s = kernels.stereo_sgbm(left, right, mode='sgbm_3way')
    def run(i):
        if i % 2:
            return np.array_equal(kernels.gaussian_blur(image, 9, 2, workspace=gaussian), expected_g)
        return np.array_equal(kernels.stereo_sgbm(left, right, mode='sgbm_3way', workspace=stereo), expected_s)
    with ThreadPoolExecutor(8) as executor:
        assert all(executor.map(run, range(24)))


@pytest.mark.parametrize('channels', [1, 3])
def test_denoise_parallel_halos_match_opencv(channels):
    cv2 = pytest.importorskip('cv2')
    shape = (137, 43) if channels == 1 else (137, 43, channels)
    image = np.random.default_rng(8).integers(0, 256, shape, dtype=np.uint8)
    assert np.array_equal(kernels.denoise(image, 12, 7, 21),
                          cv2.fastNlMeansDenoising(image, None, 12, 7, 21))


@pytest.mark.parametrize('size', [3, 5, 9, 21])
def test_float_gaussian_matches_opencv(size):
    cv2 = pytest.importorskip('cv2')
    image = np.random.default_rng(52).random((128, 128, 3), dtype=np.float32)
    actual = kernels.gaussian_blur(image, size, 2.5)
    expected = cv2.GaussianBlur(image, (size, size), 2.5)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
