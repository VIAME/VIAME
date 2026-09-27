"""Pixel operations preserve uint16 values and output storage."""
import numpy as np
import pytest

from viame.image_kernels import (to_gray, to_rgb, swap_channels, gaussian_blur,
                                 box_blur, add_weighted, erode, dilate, normalize)


@pytest.mark.parametrize("dtype,value", [(np.uint8, 200), (np.uint16, 40000),
                                         (np.float32, 40000.5)])
def test_gray_and_rgb_preserve_dtype_and_range(dtype, value):
    rgb = np.full((4, 7, 3), value, dtype=dtype)
    gray = to_gray(rgb)
    assert gray.dtype == dtype
    np.testing.assert_allclose(gray, rgb[..., 0], rtol=1e-7)
    restored = to_rgb(gray)
    assert restored.dtype == dtype
    np.testing.assert_allclose(restored, rgb, rtol=1e-7)
    swapped = swap_channels(rgb)
    assert swapped.dtype == dtype
    np.testing.assert_array_equal(swapped, rgb[..., ::-1])


@pytest.mark.parametrize("operation", [
    lambda image: gaussian_blur(image, 3),
    lambda image: box_blur(image, 3),
    lambda image: add_weighted(image, .5, image, .5),
    lambda image: erode(image),
    lambda image: dilate(image),
])
def test_uint16_operations_preserve_dtype_and_range(operation):
    image = np.full((6, 7, 3), 40000, dtype=np.uint16)
    out = operation(image)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, image)


def test_uint16_gaussian_preserves_integer_rounding():
    image = np.zeros((5, 5), dtype=np.uint16)
    image[2, 2] = 64000
    expected = np.zeros_like(image)
    expected[1:4, 1:4] = [[4000, 8000, 4000], [8000, 16000, 8000],
                         [4000, 8000, 4000]]
    actual = gaussian_blur(image, 3)
    assert actual.dtype == image.dtype
    np.testing.assert_array_equal(actual, expected)


def test_uint16_normalize_preserves_dtype():
    image = np.array([[0, 10000, 20000]], dtype=np.uint16)
    actual = normalize(image, 0, 60000)
    assert actual.dtype == image.dtype
    np.testing.assert_array_equal(actual, [[0, 30000, 60000]])
