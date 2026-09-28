# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""`imageops.convert_colour`, the array helpers, and `utilities.clustering`.

These are the functions the vendored packages call in place of a third-party
imaging library. Each was measured against the recordings when it was written;
what is pinned here is the behaviour a reader would get wrong.
"""

import numpy as np
import pytest

from viame.utilities import clustering, imageops


def _frame(seed=1, shape=(12, 16, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Channel order
# ---------------------------------------------------------------------------

def test_swapping_channels_leaves_alpha_where_it_is():
    """Reversing all four planes turns RGBA into ABGR.

    Which is not a channel order anything holds, and it corrupted every four
    channel image that went through the array version -- the C++ kernel had
    always reversed only the colour triple.
    """
    rgba = np.array([[[10, 20, 30, 40]]], dtype=np.uint8)
    assert imageops.swap_channels(rgba)[0, 0].tolist() == [30, 20, 10, 40]

    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    assert imageops.swap_channels(rgb)[0, 0].tolist() == [30, 20, 10]

    # And it is still its own inverse, on either shape.
    for image in (rgb, rgba):
        assert np.array_equal(
            imageops.swap_channels(imageops.swap_channels(image)), image)


def test_swapping_channels_agrees_with_the_kernel():
    from viame import image_kernels

    rng = np.random.default_rng(2)

    for planes in (3, 4):
        frame = rng.integers(0, 256, (8, 9, planes), dtype=np.uint8)
        assert np.array_equal(imageops.swap_channels(frame),
                              np.asarray(image_kernels.swap_channels(frame)))


def test_a_single_plane_image_passes_through_unswapped():
    grey = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    assert np.array_equal(imageops.swap_channels(grey), grey)


# ---------------------------------------------------------------------------
# Colour spaces, by name
# ---------------------------------------------------------------------------

def test_bgr_is_a_channel_order_and_naming_it_does_the_swap():
    frame = _frame()
    assert np.array_equal(imageops.convert_colour(frame, "bgr", "rgb"),
                          frame[..., ::-1])
    # And the two directions are the same operation, which is why one integer
    # code could never have distinguished them.
    assert np.array_equal(imageops.convert_colour(frame, "bgr", "rgb"),
                          imageops.convert_colour(frame, "rgb", "bgr"))


def test_the_source_order_changes_the_answer():
    """`bgr -> gray` is not `rgb -> gray`: the luma weights go by channel."""
    frame = _frame()
    assert not np.array_equal(imageops.convert_colour(frame, "bgr", "gray"),
                              imageops.convert_colour(frame, "rgb", "gray"))
    # The first is the second of the reversed frame, which is what "the name
    # says which order the caller holds" means in practice.
    assert np.array_equal(
        imageops.convert_colour(frame, "bgr", "gray"),
        imageops.convert_colour(frame[..., ::-1], "rgb", "gray"))


def test_every_space_the_vendored_packages_ask_for_is_there():
    frame = _frame()

    for space in ("bgr", "gray", "hsv", "hsv_full", "hls", "lab", "luv",
                  "xyz", "cie", "ycrcb", "yuv"):
        there = imageops.convert_colour(frame, "rgb", space)
        assert there.shape == (frame.shape[:2] if space in ("gray", "grey")
                               else frame.shape)


#: The worst a round trip through each space moves a byte, as measured.
#:
#: `bgr` is a reordering and loses nothing. The rest quantise, and the numbers
#: are not interchangeable: `ycrcb` is a linear transform in 14-bit fixed point
#: and comes back within a count, while `xyz` and `yuv` reach 21 at the gamut's
#: corners because an 8-bit intermediate cannot hold them -- white already
#: saturates two of the three XYZ planes on the way out. `hsv_full` is looser
#: than `hsv` for the reason finding 2.73 records: its forward direction
#: spreads the hue over 256 and its inverse reads it back over 255, so the pair
#: is not an involution even in principle.
#:
#: The **mean** error is under three quarters of a count for every one of them,
#: which is the number that says these are quantisation and not a mistake.
_ROUND_TRIP = {"bgr": 0, "ycrcb": 1, "hsv": 4, "hls": 5, "hsv_full": 7,
               "lab": 13, "xyz": 21, "yuv": 21}


def test_a_round_trip_through_each_space_recovers_the_frame():
    frame = _frame(shape=(24, 32, 3))

    # Grey is excluded: it throws two channels away, which is the point of it.
    for space, allowed in _ROUND_TRIP.items():
        there = imageops.convert_colour(frame, "rgb", space)
        back = imageops.convert_colour(there, space, "rgb")
        error = np.abs(back.astype(int) - frame.astype(int))
        assert error.max() <= allowed, space
        assert error.mean() < 0.75, space


def test_the_same_space_twice_is_a_no_op():
    frame = _frame()
    assert np.array_equal(imageops.convert_colour(frame, "hsv", "hsv"), frame)


def test_an_unknown_space_names_itself():
    with pytest.raises(ValueError) as raised:
        imageops.convert_colour(_frame(), "rgb", "cmyk")
    assert "cmyk" in str(raised.value)


def test_yuv_and_ycrcb_are_different_scalings_of_the_same_pair():
    frame = _frame()
    yuv = imageops.convert_colour(frame, "rgb", "yuv")
    ycrcb = imageops.convert_colour(frame, "rgb", "ycrcb")
    assert np.array_equal(yuv[..., 0], ycrcb[..., 0]), "the luma is the same"
    assert not np.array_equal(yuv[..., 1], ycrcb[..., 1])


def test_the_full_hue_range_is_a_different_conversion():
    frame = _frame()
    assert not np.array_equal(imageops.convert_colour(frame, "rgb", "hsv"),
                              imageops.convert_colour(frame, "rgb",
                                                      "hsv_full"))


# ---------------------------------------------------------------------------
# The array helpers
# ---------------------------------------------------------------------------

def test_flip_says_which_axis_without_a_table():
    frame = _frame()
    assert np.array_equal(imageops.flip(frame, vertical=True), frame[::-1])
    assert np.array_equal(imageops.flip(frame, horizontal=True),
                          frame[:, ::-1])
    assert np.array_equal(imageops.flip(frame, True, True),
                          frame[::-1, ::-1])
    assert np.array_equal(imageops.flip(frame), frame)


def test_a_lookup_table_may_be_one_curve_or_one_per_plane():
    frame = _frame()
    inverting = np.arange(255, -1, -1, dtype=np.uint8)
    assert np.array_equal(imageops.apply_lut(frame, inverting), 255 - frame)

    per_plane = np.stack([inverting, np.arange(256, dtype=np.uint8),
                          np.zeros(256, dtype=np.uint8)], axis=-1)
    out = imageops.apply_lut(frame, per_plane.reshape(1, 256, 3))
    assert np.array_equal(out[..., 0], 255 - frame[..., 0])
    assert np.array_equal(out[..., 1], frame[..., 1])
    assert (out[..., 2] == 0).all()


def test_a_lookup_table_has_256_entries():
    with pytest.raises(ValueError):
        imageops.apply_lut(_frame(), np.zeros(64, dtype=np.uint8))


def test_the_euclidean_distance_is_not_the_chamfer_one():
    """`image_kernels.distance_transform` is a three-by-three chamfer window,
    which is the usual thing called an L2 transform and is not Euclidean. The
    two differ by a fifth of a pixel, and the exact one is exact."""
    from viame import image_kernels

    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[4, 4] = 1
    mask = 1 - mask

    exact = imageops.euclidean_distance(mask)
    chamfer = np.asarray(image_kernels.distance_transform(mask))

    # The corner of a 4 by 4 offset is 4*sqrt(2).
    assert exact[0, 0] == pytest.approx(np.hypot(4, 4), abs=1e-5)
    assert abs(float(chamfer[0, 0]) - np.hypot(4, 4)) > 0.05


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def test_kmeans_is_a_function_of_its_seed():
    data = (np.random.default_rng(3).random((200, 3)) * 255).astype(np.float32)

    first = clustering.kmeans(data, 5, seed=1, max_iterations=20, epsilon=1.0)
    again = clustering.kmeans(data, 5, seed=1, max_iterations=20, epsilon=1.0)

    assert np.array_equal(first[1], again[1])
    assert first[0] == again[0]
    assert first[1].shape == (200,) and first[2].shape == (5, 3)


def test_kmeans_finds_the_clusters_that_are_there():
    rng = np.random.default_rng(4)
    middles = np.array([[10.0, 10.0], [200.0, 200.0], [10.0, 200.0]])
    data = np.concatenate([middle + rng.normal(0, 3.0, (80, 2))
                           for middle in middles]).astype(np.float32)

    _compactness, labels, centres = clustering.kmeans(data, 3, seed=7)

    # Every point in a group gets the same label, whichever number it is.
    for group in range(3):
        assert len(set(labels[group * 80:(group + 1) * 80].tolist())) == 1

    found = sorted(tuple(np.round(c, 0)) for c in centres)
    wanted = sorted(tuple(np.round(m, 0)) for m in middles)
    assert np.allclose(found, wanted, atol=2.0)


def test_kmeans_refuses_more_clusters_than_samples():
    with pytest.raises(ValueError):
        clustering.kmeans(np.zeros((3, 2)), 5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
