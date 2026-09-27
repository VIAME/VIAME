"""Blob detection, held to what it must find and what it must refuse.

This replaces `cv::SimpleBlobDetector`, which the dot calibration target
detector is built on. Unlike the calibration itself there is a right answer
to check against: the scenes here are drawn, so where every disc is and how
big it is are known exactly.

Measured against cv2 when written, on the two recorded fixtures and three
drawn scenes:

    dot grid, 35 dots      35 found, centres identical to 0.0000 px
    chessboard, 12 squares 12 found, identical -- and these are found as the
                           **holes** in the lighter region around them,
                           which is why every border is traced and not only
                           the outer ones
    discs with a streak    8 found of 8, the streak and the crescent refused
    and a crescent         by the inertia and convexity tests respectively
    the same under noise   unchanged at sigma 3 and sigma 8
"""
import numpy as np
import pytest

from viame.utilities import blobs


# What `ocv_detect_calibration_targets` asks for, so the tests exercise the
# settings that actually ship rather than the library defaults.
SETTINGS = dict(min_area=30.0, max_area=5000.0, min_circularity=0.65,
                min_inertia=0.40, min_convexity=0.70,
                min_threshold=40.0, max_threshold=220.0,
                threshold_step=10.0, min_repeatability=2)


def _disc(image, x, y, radius, value=235):
    rows, columns = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    image[((columns - x) ** 2 + (rows - y) ** 2) <= radius * radius] = value
    return image


def _grid(columns=5, rows=4, step=70, margin=50, radius=12):
    image = np.full((rows * step + 2 * margin, columns * step + 2 * margin),
                    20, np.uint8)

    truth = []
    for j in range(rows):
        for i in range(columns):
            x, y = margin + i * step, margin + j * step
            _disc(image, x, y, radius)
            truth.append((x, y))

    return np.ascontiguousarray(image), np.array(truth, dtype=np.float64)


def _matched(found, truth):
    """The worst distance from each true centre to the nearest found one."""
    assert len(found) == len(truth), (
        f"found {len(found)} blobs, drew {len(truth)}")

    gaps = np.linalg.norm(found[:, None, :] - truth[None, :, :], axis=2)

    assert len(set(gaps.argmin(axis=1))) == len(found), (
        "two blobs matched the same drawn disc")

    return gaps.min(axis=1).max()


def test_a_grid_of_discs_is_found_where_it_was_drawn():
    image, truth = _grid()

    found, diameters = blobs.detect_blobs(image, **SETTINGS)

    assert _matched(found, truth) < 0.5
    assert np.allclose(diameters, 24.0, atol=2.0)


@pytest.mark.parametrize("noise", [2.0, 5.0, 9.0])
def test_noise_does_not_move_the_centres(noise):
    """The threshold ladder is what makes this hold.

    A single threshold moves with the noise; a blob that survives a dozen of
    them and is averaged over all of them does not.
    """
    image, truth = _grid()

    rng = np.random.default_rng(int(noise))
    noisy = np.clip(image.astype(float) + rng.normal(0.0, noise, image.shape),
                    0, 255).astype(np.uint8)

    found, _ = blobs.detect_blobs(np.ascontiguousarray(noisy), **SETTINGS)

    assert _matched(found, truth) < 1.0


def test_a_streak_is_refused():
    """Smooth, and so perfectly circular by the perimeter test.

    This is the one the inertia test is for: a long thin shape has a boundary
    no more ragged than a disc's, so circularity passes it and only the ratio
    of its principal moments says no.
    """
    image = np.full((200, 300), 20, np.uint8)
    image[100:108, 60:220] = 235

    found, _ = blobs.detect_blobs(np.ascontiguousarray(image), **SETTINGS)

    assert len(found) == 0


def test_a_crescent_is_refused():
    """Round and not convex, which is what the convexity test is for.

    Two touching dots make the same shape, and taking their combined centroid
    for a dot is how a calibration ends up subtly wrong rather than obviously
    broken.
    """
    image = np.full((200, 200), 20, np.uint8)
    _disc(image, 100, 100, 30)
    _disc(image, 112, 100, 27, value=20)

    found, _ = blobs.detect_blobs(np.ascontiguousarray(image), **SETTINGS)

    assert len(found) == 0


def test_a_dark_shape_is_found_as_a_hole():
    """Blobs are lighter than their surroundings, and holes are how.

    A dark disc on a light ground is not the outer border of anything -- it
    is the hole in the region around it, and tracing only outer borders finds
    nothing at all here. The chessboard in the golden recording is exactly
    this case, twelve times over.
    """
    image = np.full((200, 200), 235, np.uint8)
    _disc(image, 100, 100, 20, value=20)

    found, _ = blobs.detect_blobs(np.ascontiguousarray(image), **SETTINGS)

    assert len(found) == 1
    assert abs(found[0][0] - 100.0) < 1.0
    assert abs(found[0][1] - 100.0) < 1.0


def test_area_bounds_are_half_open():
    """Inclusive below and exclusive above, as OpenCV's are.

    Worth a test because it is the one asymmetry a reader will assume away,
    and because a caller setting `max_area` to a disc's exact area gets
    nothing back.
    """
    image = np.full((200, 200), 20, np.uint8)
    _disc(image, 100, 100, 20)

    at_area = blobs.detect_blobs(np.ascontiguousarray(image),
                                 **{**SETTINGS, "min_area": 30.0,
                                    "max_area": 5000.0})[0]
    assert len(at_area) == 1

    area = float(np.pi * 20 * 20)

    # The bound sitting just under the disc's area refuses it; just over
    # keeps it.
    assert len(blobs.detect_blobs(np.ascontiguousarray(image),
                                  **{**SETTINGS,
                                     "max_area": area * 0.9})[0]) == 0
    assert len(blobs.detect_blobs(np.ascontiguousarray(image),
                                  **{**SETTINGS,
                                     "max_area": area * 1.2})[0]) == 1


def test_repeatability_rejects_a_one_threshold_blob():
    """A shape that only appears at one level of the ladder is not a blob."""
    image, truth = _grid()

    # Every disc at full contrast survives the whole ladder, so demanding
    # more than the ladder is long is the way to check the rule bites.
    found, _ = blobs.detect_blobs(
        image, **{**SETTINGS, "min_repeatability": 100})

    assert len(found) == 0


def test_a_filter_can_be_turned_off():
    """`None` is the cleared `filterBy...` flag."""
    image = np.full((200, 300), 20, np.uint8)
    image[100:108, 60:220] = 235

    assert len(blobs.detect_blobs(np.ascontiguousarray(image),
                                  **SETTINGS)[0]) == 0

    loose = {**SETTINGS, "min_inertia": None, "min_circularity": None,
             "min_convexity": None}
    assert len(blobs.detect_blobs(np.ascontiguousarray(image),
                                  **loose)[0]) == 1


def test_a_blank_image_finds_nothing():
    found, diameters = blobs.detect_blobs(
        np.full((150, 150), 30, np.uint8), **SETTINGS)

    assert len(found) == 0
    assert len(diameters) == 0


def test_rejects_bad_arguments():
    image, _ = _grid()

    with pytest.raises(ValueError):
        blobs.detect_blobs(np.dstack([image] * 3), **SETTINGS)

    with pytest.raises(ValueError):
        blobs.detect_blobs(image, **{**SETTINGS, "threshold_step": 0.0})
