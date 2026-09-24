"""The python bindings run the same kernels the C++ pipelines run.

That is the whole point of them: a frame resized in python and a frame
resized in a pipeline should agree. Measured against cv2 when written, on a
natural frame:

    to_gray        max difference 0   -- bit identical
    swap_channels  identical
    crop           identical
    resize         max difference 25  -- a different pixel centre convention

The resize difference is expected and is why these exist rather than Pillow:
matching the C++ half matters, matching OpenCV does not, and OpenCV is what
is being removed.
"""
import numpy as np
import pytest

from viame.image_kernels import crop, resize, swap_channels, to_gray, to_rgb


def _frame(width=64, height=48):
    y, x = np.mgrid[0:height, 0:width]
    return np.stack([(x * 4) % 256, (y * 5) % 256, ((x + y) * 3) % 256],
                    axis=-1).astype(np.uint8)


def test_resize_gives_the_requested_size():
    out = resize(_frame(), 32, 24)
    assert out.shape == (24, 32, 3)


def test_resize_of_a_flat_image_is_flat():
    flat = np.full((20, 20, 3), 77, dtype=np.uint8)
    out = resize(flat, 10, 10)
    assert out.shape == (10, 10, 3)
    assert out.min() == 77 and out.max() == 77


def test_crop_matches_a_numpy_slice():
    image = _frame()
    assert np.array_equal(crop(image, 10, 5, 20, 15), image[5:20, 10:30])


def test_crop_clamps_to_the_image():
    image = _frame(20, 20)
    out = crop(image, 15, 15, 100, 100)
    assert out.shape[0] <= 5 and out.shape[1] <= 5


def test_to_gray_uses_the_expected_weights():
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)
    image[0, 1] = (0, 255, 0)
    image[0, 2] = (0, 0, 255)
    assert list(to_gray(image)[0]) == [76, 150, 29]


def test_to_gray_needs_three_channels():
    with pytest.raises(Exception):
        to_gray(np.zeros((4, 4), dtype=np.uint8))


def test_swap_channels_is_its_own_inverse():
    image = _frame()
    assert np.array_equal(swap_channels(swap_channels(image)), image)
    assert np.array_equal(swap_channels(image)[..., 0], image[..., 2])


def test_to_rgb_repeats_the_single_channel():
    gray = to_gray(_frame())
    rgb = to_rgb(gray)
    assert rgb.shape == gray.shape + (3,)
    assert np.array_equal(rgb[..., 0], rgb[..., 1])


def test_a_two_dimensional_image_stays_two_dimensional():
    gray = to_gray(_frame())
    assert resize(gray, 16, 12).shape == (12, 16)
