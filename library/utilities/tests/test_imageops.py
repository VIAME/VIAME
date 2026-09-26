"""`imageops` is what replaced OpenCV for the primitives, so its conventions
are the contract: get them wrong and every ported call site shifts.

These assert the conventions directly rather than against cv2, since the
point of the module is that cv2 is not installed.
"""
import numpy as np
import pytest

from viame.utilities import imageops


def test_gray_uses_the_bt601_weights():
    """The weights OpenCV's COLOR_*2GRAY uses, so ported code does not shift.

    Measured against cv2 when this was written: max difference 1 on random
    input, from rounding alone.
    """
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)
    image[0, 1] = (0, 255, 0)
    image[0, 2] = (0, 0, 255)
    gray = imageops.to_gray(image)
    assert list(gray[0]) == [76, 150, 29]        # 0.299, 0.587, 0.114 * 255


def test_gray_passes_through_and_rgb_round_trips():
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert imageops.to_gray(gray) is not None
    assert np.array_equal(imageops.to_gray(gray), gray)
    rgb = imageops.to_rgb(gray)
    assert rgb.shape == (3, 4, 3)
    assert np.array_equal(rgb[..., 0], rgb[..., 2])


def test_swap_channels_is_its_own_inverse():
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert np.array_equal(imageops.swap_channels(imageops.swap_channels(image)), image)
    assert np.array_equal(imageops.swap_channels(image)[..., 0], image[..., 2])


def test_area_resize_averages_its_source_pixels():
    """`area` is the one interpolation that matched cv2 exactly (max 1), so
    it is the safe choice wherever a call site is downsampling."""
    image = np.zeros((4, 4), dtype=np.uint8)
    image[:2, :2] = 100
    out = imageops.resize(image, 2, 2, imageops.INTER_AREA)
    assert out.shape == (2, 2)
    assert out[0, 0] == 100 and out[1, 1] == 0


def test_resize_rejects_an_unknown_filter():
    with pytest.raises(ValueError):
        imageops.resize(np.zeros((4, 4), np.uint8), 2, 2, "sinc")


def test_round_trip_through_a_file(tmp_path):
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    path = tmp_path / "x.png"
    imageops.write_image(path, image)
    assert np.array_equal(imageops.read_image(path), image)
    assert imageops.read_image(path, grayscale=True).ndim == 2


@pytest.mark.parametrize("dtype,channels", [(np.uint16, 1), (np.uint8, 4)])
def test_png_roundtrip_preserves_depth_and_alpha(tmp_path, dtype, channels):
    from viame.utilities import imageops
    shape = (5, 7) if channels == 1 else (5, 7, channels)
    image = (np.arange(np.prod(shape)).reshape(shape) * 109).astype(dtype)
    path = tmp_path / "roundtrip.png"
    assert imageops.write_image(path, image) is True
    restored = imageops.read_unchanged(path)
    assert restored.dtype == image.dtype
    np.testing.assert_array_equal(restored, image)
    restored = imageops.decode_unchanged(imageops.encode_image(image))
    assert restored.dtype == image.dtype
    np.testing.assert_array_equal(restored, image)


def test_decoded_images_can_be_drawn_on(tmp_path):
    from viame.utilities import imageops
    from viame import image_kernels
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    path = tmp_path / "writable.png"
    imageops.write_image(path, image)
    data = imageops.encode_image(image)
    for restored in (imageops.read_image(path), imageops.read_unchanged(path),
                     imageops.decode_image(data), imageops.decode_unchanged(data)):
        assert restored.flags.writeable
        image_kernels.fill_polygon(restored, [(1, 1), (6, 1), (6, 6)], (255, 0, 0))
        assert restored.sum() > 0
