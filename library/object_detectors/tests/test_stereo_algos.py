# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The GMM motion detector and the geometry it hangs off.

`stereo_algos` had no tests at all before it came off cv2, which for a file
three shipped pipelines reach is the part worth fixing alongside. What is
pinned here is what the module's own docstrings claimed and nothing else had
ever checked: the hull, the oriented box, its corner order, and that the
detector finds a thing that moves and not a thing that does not.
"""

import numpy as np
import pytest

from viame.object_detectors.stereo_algos import (DetectedObject,
                                                 GMMForegroundObjectDetector)


def _bar():
    """The 5 by 2 bar the module's docstrings are written against."""
    mask = np.zeros((11, 11), dtype=np.uint8)
    mask[3:5, 2:7] = 1
    return DetectedObject.from_connected_component(mask)


def test_the_hull_is_the_four_corners_of_the_bar():
    """Andrew's monotone chain starts at the leftmost point where OpenCV's
    did not, so the run begins elsewhere; it is the same four points, and
    only `oriented_bbox` reads it, which cannot tell."""
    assert _bar().hull().tolist() == [[[2, 3]], [[6, 3]], [[6, 4]], [[2, 4]]]


def test_the_oriented_box_is_opencvs_naming_of_it():
    box = _bar().oriented_bbox()
    assert box.center == (4.0, 3.5)
    assert box.extent == (1.0, 4.0)
    assert box.angle == -90.0


def test_the_box_corners_are_in_opencvs_order():
    """`box_points()[0]` and `[2]` are a diagonal, and which diagonal decides
    the head and tail proxies a measurement pipeline uses -- so the order is
    reproduced from `cv2.boxPoints`' own formula rather than taken from
    whichever corner the minimum-area search happened to start at."""
    _, o = 0, 1
    mask = np.array([[_, _, _, o, _, _],
                     [_, _, o, o, o, _],
                     [_, o, o, o, o, o],
                     [o, o, o, o, o, _],
                     [_, o, o, o, _, _],
                     [_, _, o, o, _, _]], dtype=np.uint8)
    points = DetectedObject.from_connected_component(mask).box_points()
    assert np.allclose(points, [[2.5, 5.5], [0.0, 3.0], [3.0, 0.0], [5.5, 2.5]])


def test_the_two_box_diagonals_are_opposite_corners():
    points = _bar().box_points()
    centre = points.mean(axis=0)
    assert np.allclose((points[0] + points[2]) / 2, centre)
    assert np.allclose((points[1] + points[3]) / 2, centre)


def test_num_pixels_counts_the_mask():
    assert _bar().num_pixels() == 10


def _sequence(frames=22, height=220, width=380, moving=True):
    """Noise, and optionally a long thin thing crossing it.

    The shipped filter wants at least 800 pixels, an aspect ratio between 3.5
    and 7.5, and twelve pixels of clearance from the edge, so a target that
    does not meet all three is filtered away and the sequence proves nothing.
    An earlier version of this file used a small blob and asserted that the
    old and new code agreed -- which they did, on zero detections.
    """
    rng = np.random.default_rng(5)
    base = rng.random((height, width)) * 60 + 90
    out = []

    for t in range(frames):
        frame = base + rng.integers(-3, 4, (height, width))

        if moving and t >= 3:
            # a filled ellipse of about a thousand pixels, aspect near six,
            # stepping far enough each frame not to be learnt as background
            centre_x = 70 + 30 * ((t - 3) % 9)
            ys, xs = np.mgrid[0:height, 0:width]
            inside = (((xs - centre_x) / 50.0) ** 2 +
                      ((ys - 80) / 8.0) ** 2) <= 1.0
            frame[inside] = 235

        out.append(np.clip(frame, 0, 255).astype(np.uint8))

    return out


def test_the_detector_finds_the_thing_that_moves():
    detector = GMMForegroundObjectDetector()
    found = 0

    for frame in _sequence():
        found += len(detector.detect(frame))

    assert found > 0


def test_the_detector_finds_nothing_in_a_still_scene():
    """Noise alone is what the mixture learns, so nothing stands out of it."""
    detector = GMMForegroundObjectDetector()
    total = 0

    for frame in _sequence(moving=False):
        total += len(detector.detect(frame))

    assert total == 0


def test_the_target_is_one_the_shipped_filter_accepts():
    """Guards the sequence itself: a test that runs the detector over frames
    it rejects is a test that passes whatever the detector does."""
    detector = GMMForegroundObjectDetector()
    assert sum(len(detector.detect(f)) for f in _sequence()) >= 1


def test_the_first_frames_report_nothing_while_the_model_learns():
    """`n_startup_frames` is three, and until then the answer is empty
    whatever the mixture thinks."""
    detector = GMMForegroundObjectDetector()
    frames = _sequence()

    for frame in frames[:3]:
        assert detector.detect(frame) == []


def test_the_detector_is_deterministic():
    first = GMMForegroundObjectDetector()
    second = GMMForegroundObjectDetector()
    frames = _sequence()

    for frame in frames:
        a = [tuple(np.round(d.bbox.coords, 6)) for d in first.detect(frame)]
        b = [tuple(np.round(d.bbox.coords, 6)) for d in second.detect(frame)]
        assert a == b
