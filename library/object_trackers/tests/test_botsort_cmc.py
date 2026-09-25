# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""BoT-SORT's camera motion compensation, off cv2.

The class estimates the homography between two frames from Shi-Tomasi
corners followed by pyramidal Lucas-Kanade, and what it is worth is decided
by whether the homography it returns is the motion that was there. So these
build a pair with a known motion and check the answer against it, rather
than against a recording -- there has never been one for this tracker.
"""

import numpy as np
import pytest

from viame.image_kernels import gaussian_blur
from viame.object_trackers.pytorch.botsort_tracker import \
    CameraMotionCompensation


def _field(height=180, width=240, margin=16):
    """A textured field big enough to crop a moved window out of."""
    rng = np.random.default_rng(5)
    raw = (rng.random((height + 2 * margin, width + 2 * margin)) * 255)
    return gaussian_blur(raw.astype(np.float32), 5, 1.5)


def _pair(shift_x=0, shift_y=0, height=180, width=240, margin=16):
    field = _field(height, width, margin)
    first = np.ascontiguousarray(
        field[margin:margin + height, margin:margin + width].astype(np.uint8))
    second = np.ascontiguousarray(
        field[margin + shift_y:margin + shift_y + height,
              margin + shift_x:margin + shift_x + width].astype(np.uint8))
    return first, second


def test_the_first_frame_has_no_motion_to_report():
    cmc = CameraMotionCompensation()
    first, _ = _pair()
    assert np.array_equal(cmc.compute_homography(first), np.eye(3))


@pytest.mark.parametrize("shift_x,shift_y", [(0, 0), (4, 0), (0, 3), (5, 3)])
def test_a_translation_comes_back_as_that_translation(shift_x, shift_y):
    """A second frame cropped `shift` to the right and down shows the scene
    having moved that far the other way, so the homography from the first to
    the second is a translation by minus the shift."""
    cmc = CameraMotionCompensation()
    first, second = _pair(shift_x, shift_y)

    cmc.compute_homography(first)
    homography = cmc.compute_homography(second)

    assert homography.shape == (3, 3)

    # Where the middle of the frame goes
    middle = np.array([120.0, 90.0, 1.0])
    moved = homography @ middle
    moved = moved[:2] / moved[2]

    assert abs(moved[0] - (120.0 - shift_x)) < 0.5
    assert abs(moved[1] - (90.0 - shift_y)) < 0.5


def test_a_three_channel_frame_is_accepted():
    cmc = CameraMotionCompensation()
    first, second = _pair(4, 2)
    colour_first = np.ascontiguousarray(np.dstack([first] * 3))
    colour_second = np.ascontiguousarray(np.dstack([second] * 3))

    cmc.compute_homography(colour_first)
    homography = cmc.compute_homography(colour_second)

    middle = np.array([120.0, 90.0, 1.0])
    moved = homography @ middle
    moved = moved[:2] / moved[2]
    assert abs(moved[0] - 116.0) < 0.5
    assert abs(moved[1] - 88.0) < 0.5


def test_a_frame_with_nothing_to_track_gives_the_identity():
    """Flat ground has no corners, so there are fewer than ten keypoints and
    the estimate is refused rather than invented."""
    cmc = CameraMotionCompensation()
    flat = np.full((120, 160), 90, dtype=np.uint8)

    cmc.compute_homography(flat)
    assert np.array_equal(cmc.compute_homography(flat), np.eye(3))


def test_the_grey_conversion_weights_red_as_red():
    """It used to ask cv2 for `COLOR_BGR2GRAY` on an array vital hands over
    as RGB, which read the red channel as blue. A frame that is pure red
    should come out at red's luma, not at blue's."""
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[:, :, 0] = 255

    grey = CameraMotionCompensation._gray(red)
    assert grey.shape == (8, 8)
    assert int(grey[0, 0]) == 76      # 255 * 0.299, not 255 * 0.114


def test_a_frame_that_did_not_move_reports_almost_no_motion():
    cmc = CameraMotionCompensation()
    first, _ = _pair()

    cmc.compute_homography(first)
    homography = cmc.compute_homography(first)

    assert np.allclose(homography / homography[2, 2], np.eye(3), atol=1e-3)
