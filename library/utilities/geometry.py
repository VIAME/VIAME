"""The multi-view geometry VIAME used OpenCV for.

Classical algorithms with published derivations, not OpenCV secrets:
Hartley's normalised DLT for a homography, RANSAC around it, and Zhang's
method for intrinsics. They live here so that `opencv-python-headless` is
not a dependency of the wheel.

Accuracy is the contract, not bit equality with OpenCV. The calibration
goldens check against the synthetic scene's known answers within stated
tolerances, which is what these are held to.
"""

import numpy as np

__all__ = ["find_homography", "apply_homography"]


def _normalise(points):
    """Hartley normalisation: centre on the mean, scale to mean distance
    sqrt(2). Conditions the DLT, which is otherwise badly scaled for pixel
    coordinates, and the reason a naive DLT gives poor homographies."""
    centre = points.mean(axis=0)
    shifted = points - centre
    mean_distance = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2.0) / mean_distance if mean_distance > 1e-12 else 1.0
    transform = np.array([[scale, 0.0, -scale * centre[0]],
                          [0.0, scale, -scale * centre[1]],
                          [0.0, 0.0, 1.0]])
    return shifted * scale, transform


def _dlt(source, target):
    """The direct linear transform on four or more correspondences."""
    normalised_source, ts = _normalise(source)
    normalised_target, tt = _normalise(target)

    rows = []
    for (x, y), (u, v) in zip(normalised_source, normalised_target):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homography = vt[-1].reshape(3, 3)

    # undo the conditioning
    homography = np.linalg.inv(tt) @ homography @ ts
    if abs(homography[2, 2]) > 1e-12:
        homography = homography / homography[2, 2]
    return homography


def four_point_homography(source, target):
    """The exact homography taking four points onto four points.

    `cv2.getPerspectiveTransform`. With exactly four correspondences the
    system is determined, so there is no fitting to do and no RANSAC to run;
    this is `find_homography`'s inner solve on its own. The four points must
    be given in matching order, and no three of either set may be collinear.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)

    if len(source) != 4 or len(target) != 4:
        raise ValueError(
            "four_point_homography wants exactly four correspondences; got "
            "{} and {}".format(len(source), len(target)))

    return _dlt(source, target)


def rotation_matrix_2d(centre, angle, scale=1.0):
    """The two by three affine that rotates about `centre`.

    `cv2.getRotationMatrix2D`, including its sign convention: a positive
    `angle` is **counter-clockwise** in a coordinate system whose y runs
    down the image, which looks clockwise on screen.
    """
    radians = np.deg2rad(angle)
    alpha = scale * np.cos(radians)
    beta = scale * np.sin(radians)
    x, y = float(centre[0]), float(centre[1])

    return np.array([
        [alpha, beta, (1.0 - alpha) * x - beta * y],
        [-beta, alpha, beta * x + (1.0 - alpha) * y],
    ], dtype=np.float64)


def invert_affine(affine):
    """The inverse of a two by three affine, as a two by three.

    `cv2.invertAffineTransform`: widen to three by three with a (0, 0, 1)
    bottom row, invert, and drop the row again.
    """
    affine = np.asarray(affine, dtype=np.float64).reshape(2, 3)
    full = np.vstack([affine, [0.0, 0.0, 1.0]])
    return np.linalg.inv(full)[:2]


def fit_homography(source, target):
    """The least-squares homography over every correspondence given.

    `cv2.findHomography` with `method=0`: no RANSAC, no outlier rejection,
    every point weighted the same. That is the right call once a consensus
    set has already been chosen -- which is how the alignment uses it, to
    refit in native pixels over the inliers RANSAC found.

    Needs four correspondences, and no three of either set collinear.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)

    if len(source) != len(target):
        raise ValueError(
            "fit_homography wants matching point counts; got {} and "
            "{}".format(len(source), len(target)))

    if len(source) < 4:
        raise ValueError(
            "fit_homography needs four correspondences, got "
            "{}".format(len(source)))

    return _dlt(source, target)


def triangulate_points(projection1, projection2, points1, points2):
    """Triangulate matched points seen by two cameras.

    `cv2.triangulatePoints`, and the same direct linear transform: each
    correspondence gives four equations in the homogeneous 3D point, and the
    solution is the smallest singular vector of that four by four.

    Takes the two 3 by 4 projection matrices and two N by 2 arrays of image
    points. Returns a **4 by N** array of homogeneous points, which is the
    shape cv2 returns -- divide by the fourth row for Euclidean coordinates.
    Points at infinity come back with a fourth component near zero rather
    than as an error, which is the whole reason the answer is homogeneous.
    """
    projection1 = np.asarray(projection1, dtype=np.float64).reshape(3, 4)
    projection2 = np.asarray(projection2, dtype=np.float64).reshape(3, 4)
    points1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    points2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)

    if len(points1) != len(points2):
        raise ValueError(
            "triangulate_points wants matching point counts; got {} and "
            "{}".format(len(points1), len(points2)))

    count = len(points1)
    rows = np.empty((count, 4, 4), dtype=np.float64)

    # x * P[2] - P[0] and y * P[2] - P[1], per camera
    rows[:, 0] = points1[:, 0:1] * projection1[2] - projection1[0]
    rows[:, 1] = points1[:, 1:2] * projection1[2] - projection1[1]
    rows[:, 2] = points2[:, 0:1] * projection2[2] - projection2[0]
    rows[:, 3] = points2[:, 1:2] * projection2[2] - projection2[1]

    _, _, vt = np.linalg.svd(rows)
    return vt[:, -1, :].T


def _eight_point(source, target):
    """The fundamental matrix from eight or more correspondences.

    Hartley's normalised eight point algorithm: condition both point sets,
    solve the linear system, force rank two by zeroing the smallest singular
    value, then undo the conditioning. The rank two step is what makes it a
    fundamental matrix rather than an arbitrary three by three -- without it
    the epipolar lines do not meet at an epipole.
    """
    normalised_source, ts = _normalise(source)
    normalised_target, tt = _normalise(target)

    x1, y1 = normalised_source[:, 0], normalised_source[:, 1]
    x2, y2 = normalised_target[:, 0], normalised_target[:, 1]
    ones = np.ones(len(source))

    rows = np.column_stack(
        [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones])

    _, _, vt = np.linalg.svd(rows)
    fundamental = vt[-1].reshape(3, 3)

    # rank two
    u, singular, vt2 = np.linalg.svd(fundamental)
    singular[2] = 0.0
    fundamental = u @ np.diag(singular) @ vt2

    fundamental = tt.T @ fundamental @ ts
    norm = np.abs(fundamental).max()
    return fundamental / norm if norm > 1e-12 else fundamental


def _sampson_distance(fundamental, source, target):
    """The first-order geometric error of each correspondence.

    `cv2.findFundamentalMat`'s RANSAC threshold is on this, not on the raw
    algebraic residual -- Sampson's approximation to the distance from the
    point pair to the nearest pair that satisfies the epipolar constraint.
    """
    ones = np.ones((len(source), 1))
    p1 = np.hstack([source, ones])
    p2 = np.hstack([target, ones])

    line2 = p1 @ fundamental.T          # the epipolar line in the second view
    line1 = p2 @ fundamental            # and in the first

    residual = np.einsum("ij,ij->i", p2, line2)
    denominator = (line2[:, 0] ** 2 + line2[:, 1] ** 2 +
                   line1[:, 0] ** 2 + line1[:, 1] ** 2)
    denominator = np.where(denominator < 1e-12, 1e-12, denominator)

    return residual ** 2 / denominator


def find_fundamental(source, target, threshold=3.0, confidence=0.99,
                     max_iterations=2000, seed=0):
    """The fundamental matrix relating two views, and its inlier mask.

    `cv2.findFundamentalMat` with `FM_RANSAC`. Eight point samples, scored by
    Sampson distance -- which is what OpenCV thresholds too, so the
    `threshold` here means what `ransacReprojThreshold` meant there -- and a
    refit over the consensus set.

    Returns `(fundamental, mask)`, or `(None, None)` when there are fewer
    than eight correspondences or no consensus is found.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)

    if len(source) != len(target):
        raise ValueError("source and target must have the same length")
    if len(source) < 8:
        return None, None

    rng = np.random.default_rng(seed)
    count = len(source)
    best_inliers = np.zeros(count, dtype=bool)
    best_total = 0
    iterations = max_iterations
    squared = threshold ** 2

    step = 0
    while step < min(iterations, max_iterations):
        step += 1
        sample = rng.choice(count, 8, replace=False)
        try:
            candidate = _eight_point(source[sample], target[sample])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(candidate)):
            continue

        inliers = _sampson_distance(candidate, source, target) < squared
        total = int(inliers.sum())

        if total > best_total:
            best_total, best_inliers = total, inliers
            ratio = total / float(count)
            if ratio >= 1.0:
                break
            denominator = np.log(max(1e-12, 1.0 - ratio ** 8))
            iterations = int(np.ceil(np.log(1.0 - confidence) / denominator))

    if best_total < 8:
        return None, None

    try:
        refined = _eight_point(source[best_inliers], target[best_inliers])
    except np.linalg.LinAlgError:
        return None, None

    return refined, best_inliers


def apply_homography(homography, points):
    """Map points through a homography, returning inhomogeneous coordinates."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    mapped = homogeneous @ np.asarray(homography, dtype=np.float64).T
    w = mapped[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return mapped[:, :2] / w


def find_homography(source, target, threshold=3.0, confidence=0.995,
                    max_iterations=2000, seed=0):
    """The homography mapping `source` onto `target`, and its inlier mask.

    RANSAC over four point samples, refit on the consensus set. `threshold`
    is the reprojection distance in pixels at which a correspondence counts
    as an inlier, matching the meaning of cv2.findHomography's
    ransacReprojThreshold.

    Returns `(homography, mask)`, or `(None, None)` when there are fewer
    than four correspondences or no consensus is found.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    if len(source) != len(target):
        raise ValueError("source and target must have the same length")
    if len(source) < 4:
        return None, None
    if len(source) == 4:
        return _dlt(source, target), np.ones(4, dtype=bool)

    rng = np.random.default_rng(seed)
    count = len(source)
    best_inliers = np.zeros(count, dtype=bool)
    best_total = 0
    iterations = max_iterations

    step = 0
    while step < min(iterations, max_iterations):
        step += 1
        sample = rng.choice(count, 4, replace=False)
        try:
            candidate = _dlt(source[sample], target[sample])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(candidate)):
            continue

        error = np.sqrt(((apply_homography(candidate, source) - target) ** 2).sum(axis=1))
        inliers = error < threshold
        total = int(inliers.sum())

        if total > best_total:
            best_total, best_inliers = total, inliers
            # the standard adaptive stopping rule: once a large consensus is
            # found, the odds of a better one fall away quickly
            ratio = total / count
            if ratio > 0:
                denominator = np.log(max(1e-12, 1.0 - ratio ** 4))
                iterations = int(np.log(max(1e-12, 1.0 - confidence)) / denominator) + 1

    if best_total < 4:
        return None, None

    return _dlt(source[best_inliers], target[best_inliers]), best_inliers
