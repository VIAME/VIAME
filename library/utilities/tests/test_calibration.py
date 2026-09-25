"""Zhang's calibration, held to the rig the views are rendered through.

This replaces `cv::calibrateCamera` and `cv::stereoCalibrate`, so the
contract is accuracy on a known answer rather than agreement with OpenCV --
a calibration is a non-convex least squares problem and two implementations
that are both right land on different members of a flat minimum.

Measured against cv2 when written:

    noise free corners   exact: fx 600.0000 against a true 600, and an RMS
                         of 6e-7 where OpenCV's is 1e-5
    0.05 px of noise     fx within 4 decimal places of OpenCV's, identical
                         RMS, and the same relative error against truth
    stereo, 0.05 px      baseline, rotation, translation and RMS all equal
                         to OpenCV's to five decimal places
"""
import numpy as np
import pytest

from viame.utilities import calibration


WIDTH, HEIGHT = 640, 480
FX, FY = 600.0, 610.0
CX, CY = (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0


def _intrinsics(fx=FX, fy=FY):
    return np.array([[fx, 0.0, CX], [0.0, fy, CY], [0.0, 0.0, 1.0]])


def _board(columns=9, rows=6, square=30.0):
    gy, gx = np.mgrid[0:rows, 0:columns]
    return np.column_stack([gx.ravel() * square, gy.ravel() * square,
                            np.zeros(columns * rows)])


def _views(noise=0.0, count=12, seed=0):
    """Views spread in all three rotations and in depth.

    A calibration from views that differ only by translation cannot separate
    the focal length from the distance and comes out confidently wrong, so
    the spread is the fixture's whole point.
    """
    rng = np.random.default_rng(seed)
    board = _board()
    k = _intrinsics()

    object_points, image_points = [], []
    for index in range(count):
        rotation = np.array([0.18 * np.sin(index * 1.1),
                             0.22 * np.cos(index * 0.9),
                             0.10 * np.sin(index * 0.5)])
        translation = np.array([-120.0 + 18.0 * np.sin(index * 0.8),
                                -90.0 + 14.0 * np.cos(index * 0.7),
                                470.0 + 55.0 * np.sin(index * 0.6)])
        seen = calibration.project_points(board, rotation, translation, k,
                                          np.zeros(5))
        if noise:
            seen = seen + rng.normal(0.0, noise, seen.shape)
        object_points.append(board)
        image_points.append(seen)

    return object_points, image_points


def test_noise_free_corners_recover_the_rig_exactly():
    object_points, image_points = _views()
    rms, k, distortion, _, _ = calibration.calibrate_camera(
        object_points, image_points, (WIDTH, HEIGHT))

    assert rms < 1e-4
    assert abs(k[0, 0] - FX) / FX < 1e-5
    assert abs(k[1, 1] - FY) / FY < 1e-5
    assert abs(k[0, 2] - CX) < 1e-3
    assert abs(k[1, 2] - CY) < 1e-3

    # A *free* fit still finds a little distortion on a distortion-free rig,
    # because it has five more parameters than the data needs and float
    # rounding to spend them on -- about 1e-4 here. The golden's 1e-6 bound
    # is on a fit with the distortion flags set, which is the next test.
    assert np.abs(distortion).max() < 1e-3


def test_fixing_every_distortion_term_gives_exact_zeros():
    """Which is what the golden's 1e-6 bound on a distortion-free rig wants.

    The progressive calibration ends with all four flags set, so the shipped
    pipeline's distortion is exactly zero rather than merely small.
    """
    object_points, image_points = _views(noise=0.15)
    _, _, distortion, _, _ = calibration.calibrate_camera(
        object_points, image_points, (WIDTH, HEIGHT),
        flags=("zero_tangent_dist", "fix_k1", "fix_k2", "fix_k3"))

    assert np.abs(distortion).max() == 0.0


def test_a_noisy_calibration_is_still_within_the_golden_tolerances():
    """Focal 2%, centre 0.5% -- what `check_calibration_truth` allows."""
    object_points, image_points = _views(noise=0.15)
    _, k, _, _, _ = calibration.calibrate_camera(
        object_points, image_points, (WIDTH, HEIGHT))

    assert abs(k[0, 0] - FX) / FX < 0.02
    assert abs(k[1, 1] - FY) / FY < 0.02
    assert abs(k[0, 2] - CX) / CX < 0.005
    assert abs(k[1, 2] - CY) / CY < 0.005


def test_fixing_the_principal_point_moves_it_to_the_image_centre():
    """The flag does **not** hold the seed.

    Without `use_intrinsic_guess` OpenCV discards the seeded centre and fixes
    it at the image centre. Holding the seed instead costs half a per cent on
    the calibration fixture -- most of the tolerance it is allowed -- which
    is how this was found.
    """
    object_points, image_points = _views(noise=0.15)
    seeded = _intrinsics()
    seeded[0, 2] = 300.0
    seeded[1, 2] = 220.0

    _, k, _, _, _ = calibration.calibrate_camera(
        object_points, image_points, (WIDTH, HEIGHT),
        flags=("fix_principal_point",), intrinsics=seeded)

    assert k[0, 2] == pytest.approx(CX)
    assert k[1, 2] == pytest.approx(CY)


def test_use_intrinsic_guess_holds_the_seed_instead():
    object_points, image_points = _views(noise=0.15)
    seeded = _intrinsics()
    seeded[0, 2] = 300.0
    seeded[1, 2] = 220.0

    _, k, _, _, _ = calibration.calibrate_camera(
        object_points, image_points, (WIDTH, HEIGHT),
        flags=("fix_principal_point", "use_intrinsic_guess"),
        intrinsics=seeded)

    assert k[0, 2] == pytest.approx(300.0)
    assert k[1, 2] == pytest.approx(220.0)


def test_fixing_the_aspect_ratio_ties_fy_to_fx():
    object_points, image_points = _views(noise=0.15)
    _, k, _, _, _ = calibration.calibrate_camera(
        object_points, image_points, (WIDTH, HEIGHT),
        flags=("fix_aspect_ratio",))

    assert k[0, 0] == pytest.approx(k[1, 1])


@pytest.mark.parametrize("flag,index", [("fix_k1", 0), ("fix_k2", 1),
                                        ("fix_k3", 4)])
def test_a_fixed_distortion_term_comes_back_zero(flag, index):
    """Zero, not the seed -- the same trap as the principal point, and the
    reason a distortion-free rig was coming back with distortion on it."""
    object_points, image_points = _views(noise=0.15)
    seeded = np.full(5, 0.01)

    _, _, distortion, _, _ = calibration.calibrate_camera(
        object_points, image_points, (WIDTH, HEIGHT), flags=(flag,),
        distortion=seeded)

    assert distortion[index] == 0.0


def test_zero_tangent_dist_zeroes_both_tangential_terms():
    object_points, image_points = _views(noise=0.15)
    _, _, distortion, _, _ = calibration.calibrate_camera(
        object_points, image_points, (WIDTH, HEIGHT),
        flags=("zero_tangent_dist",), distortion=np.full(5, 0.01))

    assert distortion[2] == 0.0
    assert distortion[3] == 0.0


def test_a_non_planar_target_is_refused():
    object_points, image_points = _views()
    bent = object_points[0].copy()
    bent[0, 2] = 5.0
    object_points[0] = bent

    with pytest.raises(ValueError):
        calibration.calibrate_camera(object_points, image_points,
                                     (WIDTH, HEIGHT))


def test_mismatched_view_counts_are_refused():
    object_points, image_points = _views()
    with pytest.raises(ValueError):
        calibration.calibrate_camera(object_points, image_points[:3],
                                     (WIDTH, HEIGHT))


def _stereo_views(noise=0.05, count=14, seed=1):
    rng = np.random.default_rng(seed)
    board = _board()
    left_k, right_k = _intrinsics(600.0, 600.0), _intrinsics(610.0, 610.0)
    rig_rotation = calibration._rodrigues(np.array([0.004, -0.02, 0.001]))
    rig_translation = np.array([-120.0, 0.6, 1.4])

    object_points, left, right = [], [], []
    for index in range(count):
        rotation = np.array([0.18 * np.sin(index * 1.1),
                             0.20 * np.cos(index * 0.9),
                             0.09 * np.sin(index * 0.5)])
        translation = np.array([-120.0 + 15.0 * np.sin(index * 0.8),
                                -80.0 + 12.0 * np.cos(index * 0.7),
                                520.0 + 50.0 * np.sin(index * 0.6)])

        seen_left = calibration.project_points(board, rotation, translation,
                                               left_k, np.zeros(5))
        composed = rig_rotation @ calibration._rodrigues(rotation)
        shifted = rig_rotation @ translation + rig_translation
        seen_right = calibration.project_points(
            board, calibration._inverse_rodrigues(composed), shifted,
            right_k, np.zeros(5))

        object_points.append(board)
        left.append(seen_left + rng.normal(0.0, noise, seen_left.shape))
        right.append(seen_right + rng.normal(0.0, noise, seen_right.shape))

    return (object_points, left, right, left_k, right_k,
            rig_rotation, rig_translation)


def test_stereo_recovers_the_baseline():
    (object_points, left, right, left_k, right_k,
     rig_rotation, rig_translation) = _stereo_views()

    rms, rotation, translation, essential, fundamental = \
        calibration.stereo_calibrate(
            object_points, left, right, left_k, np.zeros(5),
            right_k, np.zeros(5), (WIDTH, HEIGHT))

    baseline = float(np.linalg.norm(translation))
    truth = float(np.linalg.norm(rig_translation))

    # The golden allows 1% on the baseline
    assert abs(baseline - truth) / truth < 0.01
    assert np.abs(rotation - rig_rotation).max() < 0.01
    assert rms < 1.0
    assert essential.shape == (3, 3)
    assert fundamental.shape == (3, 3)


def test_stereo_without_fixed_intrinsics_is_refused():
    (object_points, left, right, left_k, right_k, _, _) = _stereo_views()
    with pytest.raises(ValueError):
        calibration.stereo_calibrate(
            object_points, left, right, left_k, np.zeros(5),
            right_k, np.zeros(5), (WIDTH, HEIGHT), flags=())
