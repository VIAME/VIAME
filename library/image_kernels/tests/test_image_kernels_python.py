"""The python bindings run the same kernels the C++ pipelines run.

That is the whole point of them: a frame resized in python and a frame
resized in a pipeline should agree. Measured against cv2 when written, on a
natural frame:

    to_gray        max difference 0   -- bit identical
    swap_channels  identical
    crop           identical
    resize         max difference 25  -- a different pixel centre convention
    to_hsv/to_hls  max difference 1
    to_lab         max difference 2

The three colour spaces round trip at least as well as OpenCV's own do: on a
random frame, 4 against its 5 for HSV and HLS, and 21 for both on L*a*b*,
where the loss is the 8-bit quantisation rather than either implementation.

The resize difference is expected and is why these exist rather than Pillow:
matching the C++ half matters, matching OpenCV does not, and OpenCV is what
is being removed.
"""
import numpy as np
import pytest

from viame.image_kernels import (crop, from_hls, from_hsv, from_lab, resize,
                                 swap_channels, to_gray, to_hls, to_hsv,
                                 to_lab, to_rgb)


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


# ---------------------------------------------------------------------------
# Colour spaces
#
# Hue is on OpenCV's 0..179 scale, not 0..360, because that is what the eight
# python call sites that used `cv2.COLOR_RGB2HSV` were written against.

@pytest.mark.parametrize("forward,inverse", [(to_hsv, from_hsv),
                                             (to_hls, from_hls),
                                             (to_lab, from_lab)])
def test_a_colour_space_round_trips(forward, inverse):
    frame = _frame()
    back = inverse(forward(frame))
    assert back.shape == frame.shape
    # 8-bit L*a*b* is lossy enough that OpenCV loses as much; what is checked
    # is that nothing is grossly wrong, not that it is exact.
    assert np.abs(back.astype(int) - frame.astype(int)).max() <= 24


@pytest.mark.parametrize("convert", [to_hsv, to_hls, to_lab])
def test_a_colour_space_keeps_the_shape(convert):
    assert convert(_frame(32, 16)).shape == (16, 32, 3)


@pytest.mark.parametrize("convert", [to_hsv, to_hls, to_lab, from_hsv,
                                     from_hls, from_lab])
def test_a_colour_space_needs_three_channels(convert):
    with pytest.raises(ValueError):
        convert(np.zeros((8, 8), dtype=np.uint8))


def test_hue_of_the_primaries_is_on_the_opencv_scale():
    """Red 0, green 60 and blue 120 -- degrees halved to fit a byte."""
    primaries = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]],
                         dtype=np.uint8)
    assert list(to_hsv(primaries)[0, :, 0]) == [0, 60, 120]


def test_grey_has_no_saturation():
    grey = np.full((4, 4, 3), 128, dtype=np.uint8)
    assert to_hsv(grey)[..., 1].max() == 0
    assert to_hls(grey)[..., 2].max() == 0
