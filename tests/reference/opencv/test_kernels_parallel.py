"""The parallel kernel fast paths, against the reference implementation.

The rest of `tests/library/image_kernels/test_parallel_kernels.py` checks the
workspaces against VIAME's own single-threaded result, which needs nothing
outside this tree. These two check the numbers themselves.
"""
import numpy as np
import pytest

from viame import image_kernels as kernels


@pytest.mark.parametrize('channels', [1, 3])
def test_parallel_denoise_halos_match_the_reference(channels):
    cv2 = pytest.importorskip('cv2')
    shape = (137, 43) if channels == 1 else (137, 43, channels)
    image = np.random.default_rng(8).integers(0, 256, shape, dtype=np.uint8)
    assert np.array_equal(kernels.denoise(image, 12, 7, 21),
                          cv2.fastNlMeansDenoising(image, None, 12, 7, 21))


@pytest.mark.parametrize('size', [3, 5, 9, 21])
def test_float_gaussian_matches_the_reference(size):
    cv2 = pytest.importorskip('cv2')
    image = np.random.default_rng(52).random((128, 128, 3), dtype=np.float32)
    actual = kernels.gaussian_blur(image, size, 2.5)
    expected = cv2.GaussianBlur(image, (size, size), 2.5)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
