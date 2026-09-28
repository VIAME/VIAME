# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""`viame.utilities.cv2_api`, the surface the vendored packages import.

**These tests do not compare against cv2.** That comparison was made while the
module was written -- 96 calls of it, and the numbers are in the module's own
docstrings and in `design/lite-findings.md` -- but a test that needs cv2
installed cannot run in the install this module exists to make possible. What
is pinned here is the behaviour a reader would get wrong: the flag values,
the channel order at the file boundary, the out parameters mmcv writes
through, and the places where OpenCV's own conventions are surprising.
"""

import numpy as np
import pytest

from viame.utilities import cv2_api as cv2


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def test_the_flag_values_are_opencvs_own():
    """A config file that recorded `interpolation: 1` means bilinear by that
    number, and mmcv stores these in dictionaries, so the values are part of
    the interface and not an internal choice."""
    assert (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
            cv2.INTER_AREA, cv2.INTER_LANCZOS4) == (0, 1, 2, 3, 4)
    assert (cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101) == (0, 1, 2, 3, 4)
    assert cv2.BORDER_DEFAULT == cv2.BORDER_REFLECT_101
    assert (cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR,
            cv2.IMREAD_ANYDEPTH) == (-1, 0, 1, 2)
    assert (cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP) == (0, 1, 2)
    assert (cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE) == (1, 2)


def test_bgr_to_rgb_and_back_are_the_same_code():
    """They are, in OpenCV: both are 4, because a channel swap is its own
    inverse. Code that switches on the value cannot tell them apart and does
    not need to."""
    assert cv2.COLOR_BGR2RGB == cv2.COLOR_RGB2BGR == 4
    assert cv2.COLOR_GRAY2BGR == cv2.COLOR_GRAY2RGB == 8
    # These two are distinct, because the weights differ by channel order.
    assert cv2.COLOR_BGR2GRAY != cv2.COLOR_RGB2GRAY


# ---------------------------------------------------------------------------
# Channel order
# ---------------------------------------------------------------------------

def _frame(seed=1, shape=(12, 16, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def test_cvtcolor_names_the_order_the_caller_holds():
    frame = _frame()
    assert np.array_equal(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                          frame[..., ::-1])
    # BGR2GRAY on a frame is RGB2GRAY on its reverse, which is what "the code
    # names the caller's order" means in practice.
    assert np.array_equal(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(frame[..., ::-1], cv2.COLOR_RGB2GRAY))


def test_the_file_functions_keep_opencvs_bgr(tmp_path):
    """The one place the tree's "normalise to RGB" rule gives way.

    mmcv's contract with mmdet is BGR arrays, so `imread` returns BGR and
    `imwrite` takes it. `viame.utilities.imageops` is the RGB side of the
    same boundary, and the two must disagree by exactly a swap.
    """
    from viame.utilities import imageops

    frame = _frame()
    path = str(tmp_path / "frame.png")
    cv2.imwrite(path, frame)

    assert np.array_equal(cv2.imread(path), frame)
    assert np.array_equal(imageops.read_image(path), frame[..., ::-1])

    ok, buffer = cv2.imencode(".png", frame)
    assert ok
    assert np.array_equal(cv2.imdecode(buffer), frame)


def test_imread_of_a_missing_file_is_none_rather_than_an_error():
    """OpenCV's, and mmcv checks for it."""
    assert cv2.imread("/nonexistent/frame.png") is None
    assert cv2.imdecode(np.zeros(8, dtype=np.uint8)) is None


# ---------------------------------------------------------------------------
# The out parameters mmcv writes through
# ---------------------------------------------------------------------------

def test_cvtcolor_writes_through_its_dst():
    """`mmcv.image.imnormalize_` converts in place, on the path of every mmdet
    inference: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)`. A shim that
    returned a fresh array would leave the caller's image untouched and the
    channels swapped nowhere."""
    frame = _frame().astype(np.float32)
    original = frame.copy()
    returned = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
    assert returned is frame
    assert np.array_equal(frame, original[..., ::-1])


def test_the_arithmetic_writes_through_its_dst():
    frame = _frame().astype(np.float32)
    original = frame.copy()
    cv2.subtract(frame, np.float64([[1.0, 2.0, 3.0]]), frame)
    assert np.array_equal(frame, original - np.float32([1.0, 2.0, 3.0]))
    cv2.multiply(frame, np.float64([[2.0, 2.0, 2.0]]), frame)
    assert np.array_equal(
        frame, ((original - np.float32([1.0, 2.0, 3.0])).astype(np.float64)
                * 2.0).astype(np.float32))


def test_a_bare_number_is_every_channel_in_the_arithmetic():
    """OpenCV reads a size-one operand as one value for every channel, and it
    is the other way round for a border value -- see the next test. The two
    conventions sit a few lines apart in OpenCV and they disagree."""
    frame = np.full((4, 4, 3), 10, dtype=np.uint8)
    assert np.array_equal(cv2.add(frame, 5), np.full((4, 4, 3), 15, np.uint8))


def test_a_bare_number_is_the_first_channel_only_for_a_border():
    """`cv::Scalar(7)` is `(7, 0, 0, 0)`, so a border of `value=7` on a three
    channel image is `(7, 0, 0)` and not grey. Verified against cv2, because
    nobody writing `value=7` expects it."""
    frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    padded = cv2.copyMakeBorder(frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                value=7)
    assert list(padded[0, 0]) == [7, 0, 0]
    grey = cv2.copyMakeBorder(frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                              value=(7, 7, 7))
    assert list(grey[0, 0]) == [7, 7, 7]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_flip_follows_opencvs_sign_convention():
    frame = _frame()
    assert np.array_equal(cv2.flip(frame, 0), frame[::-1])
    assert np.array_equal(cv2.flip(frame, 1), frame[:, ::-1])
    assert np.array_equal(cv2.flip(frame, -1), frame[::-1, ::-1])


def test_resize_takes_width_then_height():
    assert cv2.resize(_frame(shape=(12, 16, 3)), (5, 9)).shape == (9, 5, 3)


def test_resize_refuses_a_factor_that_misses_whole_pixels():
    """OpenCV maps coordinates by the factor when given one and by the output
    size when given a size; the kernels only take a size, so the two agree
    only when the factor lands on an integer. Told, rather than quietly
    different."""
    frame = _frame(shape=(12, 16, 3))
    assert cv2.resize(frame, None, fx=2.0, fy=0.5).shape == (6, 32, 3)
    with pytest.raises(cv2.error):
        cv2.resize(frame, None, fx=1.3, fy=1.3)


def test_the_inverse_map_flag_is_handed_down_not_inverted_here():
    """`WARP_INVERSE_MAP` says the matrix already maps destination to source,
    which is the direction the kernel samples in. Inverting it here and
    letting the kernel invert it back costs a grey level, so the two must not
    be each other's round trip."""
    frame = _frame(shape=(24, 32, 3))
    forward = cv2.getRotationMatrix2D((16.0, 12.0), 20.0, 1.0)
    inverse = cv2.invertAffineTransform(forward)
    plain = cv2.warpAffine(frame, forward, (32, 24))
    flagged = cv2.warpAffine(frame, inverse, (32, 24),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    assert np.abs(plain.astype(int) - flagged.astype(int)).max() <= 1


def test_the_rotation_matrix_is_the_geometry_one():
    from viame.utilities import geometry

    assert np.array_equal(cv2.getRotationMatrix2D((8.0, 6.0), 30.0, 1.2),
                          geometry.rotation_matrix_2d((8.0, 6.0), 30.0, 1.2))


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_findcontours_returns_opencvs_shapes():
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[3:17, 4:20] = 1
    mask[7:13, 8:16] = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_NONE)
    assert isinstance(contours, tuple)
    assert all(c.ndim == 3 and c.shape[1] == 1 and c.shape[2] == 2
               for c in contours)
    assert all(c.dtype == np.int32 for c in contours)
    assert hierarchy.shape == (1, len(contours), 4)


def test_the_ccomp_hierarchy_names_a_holes_parent():
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[3:17, 4:20] = 1
    mask[7:13, 8:16] = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    parents = hierarchy.reshape(-1, 4)[:, 3]
    assert len(contours) == 2
    assert list(parents) == [-1, 0]
    assert (parents >= 0).any(), "the mask has a hole and the column says so"


def test_an_empty_mask_gives_no_contours_and_no_hierarchy():
    """mmdet checks `hierarchy is None`, which is what OpenCV gives here."""
    contours, hierarchy = cv2.findContours(np.zeros((8, 8), dtype=np.uint8),
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    assert contours == () and hierarchy is None


def test_connected_components_reports_areas_and_centroids():
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:6, 3:8] = 1          # 4 by 5, centred on (5.0, 3.5)
    mask[8:11, 11:15] = 1       # 3 by 4
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    assert count == 3
    assert sorted(stats[:, cv2.CC_STAT_AREA]) == [12, 20, 12 * 16 - 32]
    big = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    assert stats[big, cv2.CC_STAT_AREA] == 20
    assert tuple(centroids[big]) == (5.0, 3.5)
    assert tuple(stats[big, :4]) == (3, 2, 5, 4)


def test_boxpoints_turns_a_rectangle_back_into_corners():
    rect = ((10.0, 8.0), (6.0, 4.0), 0.0)
    corners = cv2.boxPoints(rect)
    assert corners.shape == (4, 2) and corners.dtype == np.float32
    assert sorted(map(tuple, corners)) == [(7.0, 6.0), (7.0, 10.0),
                                           (13.0, 6.0), (13.0, 10.0)]


def test_calchist_counts_bytes():
    plane = np.array([[0, 0, 7], [255, 7, 7]], dtype=np.uint8)
    histogram = cv2.calcHist([plane], [0], None, [256], [0, 256])
    assert histogram.shape == (256,) and histogram.dtype == np.float32
    assert histogram[0] == 2 and histogram[7] == 3 and histogram[255] == 1
    assert histogram.sum() == plane.size


def test_calchist_says_what_it_does_not_do():
    plane = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(cv2.error):
        cv2.calcHist([plane], [0], None, [64], [0, 256])


def test_cart_to_polar_is_opencvs_polynomial_not_arctan2():
    """They agree to a third of a degree and that is the point.

    `warpPolar` builds its map with the same approximation, and `imgaug` then
    puts keypoints through `cartToPolar` expecting them to land where the
    pixels went. Using an accurate `arctan2` in one and OpenCV's polynomial
    in the other would pull the two apart.
    """
    across = np.array([[1.0], [0.0], [-1.0], [0.0]], dtype=np.float32)
    down = np.array([[0.0], [1.0], [0.0], [-1.0]], dtype=np.float32)
    radius, angle = cv2.cartToPolar(across, down)
    assert np.allclose(radius.reshape(-1), [1.0, 1.0, 1.0, 1.0], atol=1e-6)
    assert np.allclose(angle.reshape(-1),
                       [0.0, np.pi / 2, np.pi, 3 * np.pi / 2], atol=6e-3)
    # Never negative, which is what the [0, 2pi) range means.
    assert (angle >= 0).all()


def test_convert_maps_hands_the_float_maps_back():
    """It converts nothing, and that is a difference rather than a saving:
    OpenCV quantises to a fifth of a bit of a pixel and this does not."""
    across = np.random.default_rng(1).random((4, 5)).astype(np.float32) * 10
    down = np.random.default_rng(2).random((4, 5)).astype(np.float32) * 10
    first, second = cv2.convertMaps(across, down, cv2.CV_16SC2)
    assert np.array_equal(first, across) and np.array_equal(second, down)


def test_laplacian_is_the_five_point_stencil_at_aperture_one():
    frame = np.zeros((7, 7), dtype=np.float64)
    frame[3, 3] = 1.0
    response = cv2.Laplacian(frame, cv2.CV_64F)
    assert response[3, 3] == -4.0
    assert response[2, 3] == response[4, 3] == 1.0
    assert response[3, 2] == response[3, 4] == 1.0
    # Aperture 3 puts the weight on the corners instead, which is not what
    # deriving it from the five-point stencil would give.
    corners = cv2.Laplacian(frame, cv2.CV_64F, ksize=3)
    assert corners[3, 3] == -8.0
    assert corners[2, 2] == 2.0 and corners[2, 3] == 0.0


def test_kmeans_is_reproducible_from_the_rng_seed():
    """Which is the whole contract. `imgaug`'s own comment at the call site
    says cv2's is not deterministic without `setRNGSeed` and gives no way to
    read the state back."""
    data = (np.random.default_rng(3).random((200, 3)) * 255).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 1.0)

    cv2.setRNGSeed(1)
    first = cv2.kmeans(data, 5, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    cv2.setRNGSeed(1)
    again = cv2.kmeans(data, 5, None, criteria, 1, cv2.KMEANS_PP_CENTERS)

    assert np.array_equal(first[1], again[1])
    assert first[1].shape == (200, 1) and first[2].shape == (5, 3)

    cv2.setRNGSeed(9)
    other = cv2.kmeans(data, 5, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    assert first[0] > 0.0 and other[0] > 0.0


def test_resize_takes_int32_for_nearest_neighbour_only():
    """As cv2 does: it refuses int32 for every other interpolation too."""
    labels = np.arange(48 * 64, dtype=np.int32).reshape(48, 64)
    assert cv2.resize(labels, (32, 24),
                      interpolation=cv2.INTER_NEAREST).dtype == np.int32
    with pytest.raises(cv2.error):
        cv2.resize(labels, (32, 24), interpolation=cv2.INTER_LINEAR)


def test_the_pyramid_mean_shift_refuses_the_level_it_cannot_reproduce():
    """And its default differs from cv2's, deliberately: cv2 defaults to 1."""
    frame = _frame(shape=(16, 16, 3))
    assert cv2.pyrMeanShiftFiltering(frame, 3, 20).shape == frame.shape
    with pytest.raises(cv2.error):
        cv2.pyrMeanShiftFiltering(frame, 3, 20, maxLevel=1)


def test_warp_polar_refuses_the_semi_log_mapping():
    frame = _frame(shape=(16, 16, 3))
    with pytest.raises(cv2.error):
        cv2.warpPolar(frame, (0, 0), (8.0, 8.0), 11.0,
                      cv2.WARP_POLAR_LOG | cv2.INTER_LINEAR)


def test_the_out_parameter_is_a_hint_when_the_shape_changes():
    """OpenCV's `dst` is an `OutputArray`: `imgaug` hands `cvtColor` a three
    channel buffer and asks for grey, and cv2 reallocates rather than
    failing."""
    frame = _frame(shape=(8, 8, 3))
    buffer = np.zeros((8, 8, 3), dtype=np.uint8)
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, buffer)
    assert result.shape == (8, 8)
    assert result is not buffer


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------

def test_the_windowing_functions_say_there_is_no_window():
    for call in (cv2.imshow, cv2.namedWindow, cv2.getWindowProperty):
        with pytest.raises(cv2.error):
            call("a window", 0)
    # Not an error: `imshow` has already refused, and a caller polling for a
    # keypress is honestly told nothing was pressed.
    assert cv2.waitKey(1) == -1


def test_umat_is_never_an_instance_and_cannot_be_made():
    """`imgaug.augmenters.flip` guards on `isinstance(image, cv2.UMat)` to
    decide whether an array is already on a GPU. Nothing here is, so the
    guard must take its other branch."""
    assert not isinstance(np.zeros((2, 2)), cv2.UMat)
    with pytest.raises(cv2.error):
        cv2.UMat(np.zeros((2, 2), dtype=np.uint8))


def test_every_colour_space_the_vendored_packages_ask_for_is_there():
    """`imgaug.augmenters.color` builds a table of all ten at import time and
    a `ChangeColorspace` can reach any of them."""
    frame = _frame()
    for code in (cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2LUV, cv2.COLOR_RGB2YUV,
                 cv2.COLOR_RGB2YCR_CB, cv2.COLOR_RGB2LAB, cv2.COLOR_RGB2HLS,
                 cv2.COLOR_RGB2HSV, cv2.COLOR_BGR2XYZ, cv2.COLOR_BGR2LUV,
                 cv2.COLOR_BGR2YUV, cv2.COLOR_BGR2YCR_CB):
        assert cv2.cvtColor(frame, code).shape == frame.shape
    for code in (cv2.COLOR_XYZ2RGB, cv2.COLOR_LUV2RGB, cv2.COLOR_YUV2RGB,
                 cv2.COLOR_YCR_CB2RGB, cv2.COLOR_LAB2RGB, cv2.COLOR_HLS2RGB,
                 cv2.COLOR_HSV2RGB):
        assert cv2.cvtColor(frame, code).shape == frame.shape


def test_an_unimplemented_colour_space_names_itself():
    with pytest.raises(cv2.error) as raised:
        cv2.cvtColor(_frame(), cv2.COLOR_RGB2HLS_FULL)
    assert "COLOR_RGB2HLS_FULL" in str(raised.value)


def test_setnumthreads_can_lower_the_budget_and_put_it_back():
    from viame import image_kernels

    before = image_kernels.kernel_thread_count()
    try:
        cv2.setNumThreads(1)
        assert image_kernels.kernel_thread_count() == 1
        # OpenCV's 0 means "on the calling thread", which is one worker.
        cv2.setNumThreads(0)
        assert image_kernels.kernel_thread_count() == 1
        cv2.setNumThreads(-1)
        assert image_kernels.kernel_thread_count() == before
    finally:
        image_kernels.set_kernel_thread_count(0)


def test_the_version_is_not_a_version_opencv_released():
    """`mmcv.utils.collect_env` puts it in a banner, and a banner claiming an
    OpenCV that is not installed is worse than one naming this module."""
    assert "viame" in cv2.__version__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
