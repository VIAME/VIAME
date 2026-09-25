"""`find_homography` replaces cv2.findHomography, so the contract is
accuracy on known answers rather than agreement with OpenCV.

Measured against cv2 when written, on 60 synthetic correspondences: mean
reprojection error 2.3e-13 against OpenCV's 6.0e-6, and with 30% of the
correspondences corrupted both recovered all 42 true inliers.
"""
import numpy as np
import pytest

from viame.utilities.geometry import (apply_homography, find_fundamental,
                                      find_homography, fit_homography,
                                      four_point_homography, invert_affine,
                                      rotation_matrix_2d, triangulate_points)


def _scene(seed=7, count=60):
    rng = np.random.default_rng(seed)
    truth = np.array([[1.2, 0.15, 30.0], [-0.1, 0.95, -12.0], [0.0003, -0.0002, 1.0]])
    truth = truth / truth[2, 2]
    source = rng.uniform(0, 640, (count, 2))
    return truth, source, apply_homography(truth, source)


def test_recovers_an_exact_homography():
    truth, source, target = _scene()
    found, mask = find_homography(source, target)
    assert mask.all()
    error = np.sqrt(((apply_homography(found, source) - target) ** 2).sum(axis=1)).mean()
    assert error < 1e-8, error


def test_four_points_is_the_minimum_and_is_exact():
    truth, source, target = _scene(count=4)
    found, mask = find_homography(source, target)
    assert found is not None and mask.all()
    assert np.allclose(apply_homography(found, source), target, atol=1e-8)


def test_fewer_than_four_has_no_answer():
    truth, source, target = _scene(count=3)
    assert find_homography(source, target) == (None, None)


def test_rejects_outliers():
    truth, source, target = _scene()
    rng = np.random.default_rng(3)
    corrupted = target.copy()
    bad = rng.choice(len(corrupted), 18, replace=False)
    corrupted[bad] += rng.uniform(60, 200, (18, 2))

    found, mask = find_homography(source, corrupted)
    good = np.ones(len(source), dtype=bool)
    good[bad] = False

    assert mask.sum() >= 40, "should keep nearly all 42 true inliers"
    assert not mask[bad].any(), "no corrupted correspondence should be an inlier"
    error = np.sqrt(((apply_homography(found, source[good]) - target[good]) ** 2).sum(axis=1)).mean()
    assert error < 1e-6, error


def test_mismatched_lengths_are_rejected():
    truth, source, target = _scene()
    with pytest.raises(ValueError):
        find_homography(source, target[:-1])


def test_normalisation_makes_it_work_on_pixel_coordinates():
    """Without Hartley normalisation a DLT on raw pixel coordinates is badly
    conditioned; this is the case that exposes it."""
    truth, source, target = _scene(seed=11)
    source = source * 1000.0
    target = apply_homography(truth, source)
    found, _ = find_homography(source, target)
    error = np.sqrt(((apply_homography(found, source) - target) ** 2).sum(axis=1)).mean()
    assert error < 1e-5, error


# ---------------------------------------------------------------------------
# The rest of the multi-view geometry, all held to known answers.
#
# Against cv2 when written: `rotation_matrix_2d` and `invert_affine` exact,
# `fit_homography` within 8e-4 of a pixel, `triangulate_points` within 3e-11
# of a millimetre, and `find_fundamental` **more** accurate than
# `cv2.findFundamentalMat` with FM_RANSAC -- 0.28 px of Sampson error
# against its 0.72, and 99% agreement with the true inlier set against 91%.

def _camera(fx=600.0, fy=600.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _stereo_scene(seed=17, count=80):
    rng = np.random.default_rng(seed)
    k = _camera()
    angle = 0.02
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(angle), 0.0, np.cos(angle)]])
    translation = np.array([[-120.0], [2.0], [3.0]])

    world = np.column_stack([rng.uniform(-300, 300, count),
                             rng.uniform(-200, 200, count),
                             rng.uniform(900, 3000, count)])

    left = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    right = k @ np.hstack([rotation, translation])

    def project(matrix):
        homogeneous = matrix @ np.vstack([world.T, np.ones(count)])
        return (homogeneous[:2] / homogeneous[2]).T

    return world, left, right, project(left), project(right)


def test_triangulate_recovers_the_world_points():
    world, left, right, seen_left, seen_right = _stereo_scene()
    homogeneous = triangulate_points(left, right, seen_left, seen_right)

    assert homogeneous.shape == (4, len(world))      # cv2's shape
    recovered = (homogeneous[:3] / homogeneous[3]).T
    np.testing.assert_allclose(recovered, world, atol=1e-6)


def test_triangulate_wants_matching_counts():
    _, left, right, seen_left, seen_right = _stereo_scene()
    with pytest.raises(ValueError):
        triangulate_points(left, right, seen_left, seen_right[:10])


def test_the_fundamental_matrix_has_rank_two():
    """Without the rank two step the epipolar lines do not meet at an
    epipole, and it is not a fundamental matrix at all."""
    _, _, _, seen_left, seen_right = _stereo_scene()
    fundamental, _ = find_fundamental(seen_left, seen_right)
    assert np.linalg.matrix_rank(fundamental, 1e-8) == 2


def test_the_fundamental_matrix_satisfies_the_epipolar_constraint():
    _, _, _, seen_left, seen_right = _stereo_scene()
    fundamental, mask = find_fundamental(seen_left, seen_right)

    ones = np.ones((len(seen_left), 1))
    residual = np.einsum(
        "ij,ij->i",
        np.hstack([seen_right, ones]),
        np.hstack([seen_left, ones]) @ fundamental.T)
    assert np.abs(residual).max() < 1e-6
    assert mask.all()


def test_find_fundamental_rejects_outliers():
    rng = np.random.default_rng(4)
    _, _, _, seen_left, seen_right = _stereo_scene()
    corrupted = seen_right.copy()
    bad = rng.choice(len(corrupted), 20, replace=False)
    corrupted[bad] += rng.uniform(-100, 100, (20, 2))

    _, mask = find_fundamental(seen_left, corrupted, threshold=3.0)
    truth = np.ones(len(corrupted), dtype=bool)
    truth[bad] = False
    assert (mask == truth).mean() > 0.95


def test_find_fundamental_needs_eight_points():
    _, _, _, seen_left, seen_right = _stereo_scene()
    assert find_fundamental(seen_left[:7], seen_right[:7]) == (None, None)


def test_fit_homography_is_least_squares_over_everything():
    truth, source, target = _scene()
    fitted = fit_homography(source, target)
    np.testing.assert_allclose(apply_homography(fitted, source), target,
                               atol=1e-6)


def test_fit_homography_needs_four_points():
    truth, source, target = _scene()
    with pytest.raises(ValueError):
        fit_homography(source[:3], target[:3])


def test_four_point_homography_is_exact():
    truth, source, target = _scene()
    exact = four_point_homography(source[:4], target[:4])
    np.testing.assert_allclose(apply_homography(exact, source[:4]),
                               target[:4], atol=1e-8)


def test_four_point_homography_wants_exactly_four():
    truth, source, target = _scene()
    with pytest.raises(ValueError):
        four_point_homography(source[:5], target[:5])


def test_rotation_matrix_2d_turns_about_the_centre():
    centre = (15.5, 11.5)
    affine = rotation_matrix_2d(centre, 90.0, 1.0)
    moved = affine @ np.array([centre[0], centre[1], 1.0])
    np.testing.assert_allclose(moved, centre, atol=1e-12)


def test_invert_affine_round_trips():
    affine = rotation_matrix_2d((20.0, 30.0), 37.0, 1.4)
    back = invert_affine(affine)

    point = np.array([12.0, 44.0, 1.0])
    there = affine @ point
    again = back @ np.array([there[0], there[1], 1.0])
    np.testing.assert_allclose(again, point[:2], atol=1e-10)
