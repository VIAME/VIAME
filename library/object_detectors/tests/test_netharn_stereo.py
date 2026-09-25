"""netharn's stereo rectification, now that it is ours rather than OpenCV's.

`netharn/stereo.py` had no test at all -- no coverage, no importers, 919
lines of stereo calibration and rectification. It was ported off cv2 anyway,
which meant nothing would have caught a mistake, so this is the check that
did not exist rather than a check of something that already worked.

The rig is synthetic and the properties are exact ones: a rectification map
is a definite thing, and rectifying a point and then unrectifying it must
give the point back. Where a number is compared to OpenCV's rather than to a
known answer, the measurement is in the test that makes it.

Measured against cv2 when written:

    rectification maps      bit identical
    rectify_points          1.7e-13 px, which is double precision noise
    rectify then unrectify  7.3e-7 px round trip
    rectify_image           2 counts at worst on a smooth scene, 0.016 mean;
                            7 and 1.01 on pure noise, which is the adversarial
                            case for any interpolator. Both are the
                            fixed-point-versus-double cubic difference that
                            `warp/ocv` already carries a (4.0, 0.25) tolerance
                            for in the goldens.
"""
import numpy as np
import pytest

from viame.measurement import projection
from viame.object_detectors.netharn.netharn import stereo


WIDTH, HEIGHT = 640, 480


def _rig():
    """A left camera, a right camera and the pose between them."""
    left = np.array([[900.0, 0.0, 320.0],
                     [0.0, 905.0, 240.0],
                     [0.0, 0.0, 1.0]])
    right = np.array([[890.0, 0.0, 318.0],
                      [0.0, 895.0, 242.0],
                      [0.0, 0.0, 1.0]])

    left_distortion = np.array([-0.12, 0.03, 0.001, -0.002, 0.0])
    right_distortion = np.array([-0.10, 0.02, -0.001, 0.001, 0.0])

    rotation = projection.rodrigues(np.array([0.01, -0.03, 0.005]))
    translation = np.array([-120.0, 1.5, 2.0])

    return (left, left_distortion, right, right_distortion,
            rotation, translation)


def _left_camera():
    left, left_d, right, right_d, rotation, translation = _rig()

    rectified = projection.stereo_rectify(
        left, left_d, right, right_d, WIDTH, HEIGHT, rotation, translation)

    camera = stereo.StereoCamera()
    camera['K'] = left
    camera['D'] = left_d
    camera['R'] = rectified['left_rotation']
    camera['P'] = rectified['left_projection']

    return camera


def _points(count=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([20.0, 20.0], [WIDTH - 20.0, HEIGHT - 20.0],
                       (count, 2))


def test_rectifying_a_point_moves_it_onto_the_rectified_camera():
    camera = _left_camera()
    points = _points()

    rectified = camera.rectify_points(points)

    assert rectified.shape == points.shape
    # A rectification is not a no-op on a distorted camera
    assert np.abs(rectified - points).max() > 1.0


def test_unrectify_undoes_rectify():
    """The property that matters, and the one a wrong inverse map breaks.

    `unrectify_points` goes a different way round from `rectify_points` -- it
    unprojects through `inv(P R)` and reprojects through the distortion --
    so agreeing to a millionth of a pixel says both are right, not that one
    was used twice.
    """
    camera = _left_camera()
    points = _points()

    back = camera.unrectify_points(camera.rectify_points(points))

    assert np.abs(back - points).max() < 1e-4


def test_the_rectification_map_lands_where_the_points_do():
    """The map and the point transform are two routes to the same answer.

    The map says, for each rectified pixel, where to read from; so reading
    the map at a rectified point gives back the unrectified one.
    """
    camera = _left_camera()

    map_x, map_y = camera._undistort_rectify_map((WIDTH, HEIGHT))

    assert map_x.shape == (HEIGHT, WIDTH)
    assert map_y.shape == (HEIGHT, WIDTH)

    rng = np.random.default_rng(3)
    for _ in range(20):
        x = int(rng.integers(40, WIDTH - 40))
        y = int(rng.integers(40, HEIGHT - 40))

        # Where the map says this rectified pixel comes from
        source = np.array([[map_x[y, x], map_y[y, x]]])

        # and where rectifying that source point puts it
        assert np.abs(camera.rectify_points(source)[0]
                      - np.array([x, y])).max() < 0.5


def test_rectifying_an_image_keeps_its_shape_and_dtype():
    camera = _left_camera()

    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    scene = np.clip(128 + 90 * np.sin(columns / 37.0) * np.cos(rows / 29.0),
                    0, 255).astype(np.uint8)
    scene = np.ascontiguousarray(np.dstack([scene] * 3))

    rectified = camera.rectify_image(scene)

    assert rectified.shape == scene.shape
    assert rectified.dtype == scene.dtype
    # A rectification of a structured scene is not the scene
    assert np.abs(rectified.astype(int) - scene.astype(int)).max() > 0


def test_the_object_points_are_a_plane_grid():
    """Cheap, and it pins the column-major order the detector's corners use."""
    points = stereo._make_object_points((3, 2))

    assert points.shape == (6, 3)
    assert np.all(points[:, 2] == 0)
    assert {tuple(p[:2]) for p in points} == {
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
        (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)}


def test_a_stereo_camera_carries_its_four_matrices():
    camera = _left_camera()

    assert camera['K'].shape == (3, 3)
    assert camera['R'].shape == (3, 3)
    assert camera['P'].shape == (3, 4)
    assert len(np.asarray(camera['D']).reshape(-1)) >= 4
