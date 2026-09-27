"""The camera geometry bindings, against geometry with known answers.

These are held to what the maths says rather than to a recording of OpenCV,
because that is the stronger check and it is available here: a point
projected and then unprojected comes back, a rotation and its inverse
compose to the identity, and an undistortion undoes a distortion. Measured
against cv2 when written, on a 600 by 610 pixel camera with Brown-Conrady
coefficients (-0.21, 0.05, 0.001, -0.002, 0):

    project_points      max difference 0     -- bit identical
    undistort_points    max difference 2e-12
    rodrigues           max difference 0
    stereo_rectify      max difference 1e-12
    rectification_maps  max difference 1e-5  -- the maps are float32

`tests/golden/projection` records the same functions against OpenCV; this
is the fast unit half.
"""
import numpy as np
import pytest

from viame.measurement import projection


K = np.array([[600.0, 0.0, 320.0],
              [0.0, 610.0, 240.0],
              [0.0, 0.0, 1.0]])

DISTORTION = [-0.21, 0.05, 0.001, -0.002, 0.0]


def _points_3d(n=12, seed=4):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-0.4, 0.4, n),
                            rng.uniform(-0.3, 0.3, n),
                            rng.uniform(2.0, 6.0, n)])


# ---------------------------------------------------------------------------
# Rodrigues

def test_rodrigues_gives_a_rotation_matrix():
    matrix = projection.rodrigues(np.array([0.1, -0.2, 0.05]))
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(matrix), 1.0)


def test_rodrigues_round_trips():
    vector = np.array([0.31, -0.12, 0.44])
    back = projection.rodrigues(projection.rodrigues(vector))
    np.testing.assert_allclose(back, vector, atol=1e-12)


def test_rodrigues_of_zero_is_the_identity():
    np.testing.assert_allclose(projection.rodrigues(np.zeros(3)), np.eye(3),
                               atol=1e-15)


def test_rodrigues_of_the_identity_is_zero():
    np.testing.assert_allclose(projection.rodrigues(np.eye(3)), np.zeros(3),
                               atol=1e-15)


# ---------------------------------------------------------------------------
# Projection

def test_project_points_without_distortion_is_the_pinhole():
    points = _points_3d()
    got = projection.project_points(points, K)
    expected = np.column_stack([
        K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
        K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]])
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_project_points_takes_a_rotation_as_a_vector_or_a_matrix():
    points = _points_3d()
    vector = np.array([0.05, -0.02, 0.01])
    by_vector = projection.project_points(points, K, DISTORTION, vector)
    by_matrix = projection.project_points(points, K, DISTORTION,
                                          projection.rodrigues(vector))
    np.testing.assert_allclose(by_vector, by_matrix, atol=1e-12)


def test_project_points_applies_the_translation():
    points = _points_3d()
    shift = np.array([0.1, -0.05, 0.3])
    moved = projection.project_points(points, K, [], np.eye(3), shift)
    direct = projection.project_points(points + shift, K)
    np.testing.assert_allclose(moved, direct, atol=1e-12)


def test_project_points_wants_three_wide():
    with pytest.raises(ValueError):
        projection.project_points(np.zeros((5, 2)), K)


def test_project_points_takes_the_shape_cv2_hands_back():
    """(N, 1, 3) as well as (N, 3): the call sites carry cv2's shape around."""
    points = _points_3d()
    flat = projection.project_points(points, K)
    nested = projection.project_points(points.reshape(-1, 1, 3), K)
    np.testing.assert_allclose(flat, nested)


# ---------------------------------------------------------------------------
# Undistortion

def test_undistort_undoes_the_distortion():
    points = _points_3d()
    distorted = projection.project_points(points, K, DISTORTION)
    ideal = projection.project_points(points, K)

    recovered = projection.undistort_points(distorted, K, DISTORTION, None, K)
    np.testing.assert_allclose(recovered, ideal, atol=0.05)


def test_undistort_with_no_coefficients_is_exact():
    points = _points_3d()
    projected = projection.project_points(points, K)
    recovered = projection.undistort_points(projected, K, [], None, K)
    np.testing.assert_allclose(recovered, projected, atol=1e-10)


def test_undistort_without_a_projection_gives_normalised_coordinates():
    points = _points_3d()
    projected = projection.project_points(points, K)
    normalised = projection.undistort_points(projected, K, [])
    np.testing.assert_allclose(normalised,
                               points[:, :2] / points[:, 2:3], atol=1e-10)


def test_undistort_wants_two_wide():
    with pytest.raises(ValueError):
        projection.undistort_points(np.zeros((5, 3)), K, [])


# ---------------------------------------------------------------------------
# Stereo

def _rig():
    rotation = projection.rodrigues(np.array([0.004, -0.02, 0.001]))
    translation = np.array([-120.0, 0.6, 1.4])
    return rotation, translation


def test_stereo_rectify_gives_the_five_matrices():
    rotation, translation = _rig()
    out = projection.stereo_rectify(K, DISTORTION, K, DISTORTION, 640, 480,
                                    rotation, translation)

    assert set(out) == {"left_rotation", "right_rotation", "left_projection",
                        "right_projection", "disparity_to_depth"}
    assert out["left_rotation"].shape == (3, 3)
    assert out["left_projection"].shape == (3, 4)
    assert out["disparity_to_depth"].shape == (4, 4)


def test_stereo_rectify_puts_the_principal_points_together():
    """CALIB_ZERO_DISPARITY: a correspondence becomes a horizontal shift."""
    rotation, translation = _rig()
    out = projection.stereo_rectify(K, DISTORTION, K, DISTORTION, 640, 480,
                                    rotation, translation)

    left = out["left_projection"]
    right = out["right_projection"]
    assert np.isclose(left[0, 2], right[0, 2])
    assert np.isclose(left[1, 2], right[1, 2])

    # The baseline times the focal length, negated, in the fourth column
    assert right[0, 3] < 0


def test_the_rectifying_rotations_are_rotations():
    rotation, translation = _rig()
    out = projection.stereo_rectify(K, DISTORTION, K, DISTORTION, 640, 480,
                                    rotation, translation)

    for name in ("left_rotation", "right_rotation"):
        matrix = out[name]
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# Rectification maps

def test_rectification_maps_are_the_requested_size():
    map_x, map_y = projection.rectification_maps(K, DISTORTION, np.eye(3), K,
                                                 64, 48)
    assert map_x.shape == (48, 64)
    assert map_y.shape == (48, 64)


def test_an_undistorted_identity_map_is_the_identity():
    """No distortion, no rotation and the same K: every pixel maps to itself."""
    map_x, map_y = projection.rectification_maps(K, [], np.eye(3), K, 64, 48)
    expected_x, expected_y = np.meshgrid(np.arange(64.0), np.arange(48.0))
    np.testing.assert_allclose(map_x, expected_x, atol=1e-3)
    np.testing.assert_allclose(map_y, expected_y, atol=1e-3)


def test_rectification_maps_land_inside_the_image_for_a_real_rig():
    rotation, translation = _rig()
    out = projection.stereo_rectify(K, DISTORTION, K, DISTORTION, 640, 480,
                                    rotation, translation)
    map_x, map_y = projection.rectification_maps(
        K, DISTORTION, out["left_rotation"], out["left_projection"], 640, 480)

    # An alpha of zero zooms in until there is no invalid border, so the
    # maps should be sampling real pixels nearly everywhere.
    inside = ((map_x >= 0) & (map_x < 640) & (map_y >= 0) & (map_y < 480))
    assert inside.mean() > 0.99
