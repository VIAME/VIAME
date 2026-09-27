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

    design = np.asarray(rows, dtype=np.float64)
    # Four pairs give an 8x9 system: retain the ninth null-space vector.
    # Larger fits need only thin U/V, not a quadratic (2*n)x(2*n) U matrix.
    _, singular, vt = np.linalg.svd(design, full_matrices=len(rows) < 9)
    if singular[7] <= singular[0] * 1e-12:
        raise np.linalg.LinAlgError("degenerate homography correspondences")
    homography = vt[-1].reshape(3, 3)

    # undo the conditioning
    homography = np.linalg.inv(tt) @ homography @ ts
    if abs(homography[2, 2]) > 1e-12:
        homography = homography / homography[2, 2]
    if not np.all(np.isfinite(homography)) or np.linalg.matrix_rank(homography) < 3:
        raise np.linalg.LinAlgError("singular homography")
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


def epipolar_lines(fundamental, points, which_image):
    """The epipolar lines in the other view, for each point.

    `cv2.computeCorrespondEpilines`. `which_image` is 1 when the points are
    in the first view -- giving lines in the second, as `F p` -- and 2 for
    the reverse, `F^T p`. Each line is returned as (a, b, c) normalised so
    that a squared plus b squared is one, which is what makes `a x + b y + c`
    the signed distance from the line and is how every caller uses it.
    """
    fundamental = np.asarray(fundamental, dtype=np.float64).reshape(3, 3)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    if which_image not in (1, 2):
        raise ValueError("which_image is 1 or 2, got {}".format(which_image))

    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    matrix = fundamental if which_image == 1 else fundamental.T
    lines = homogeneous @ matrix.T

    scale = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2)
    scale = np.where(scale < 1e-12, 1.0, scale)

    return lines / scale[:, None]


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
        try:
            return _dlt(source, target), np.ones(4, dtype=bool)
        except np.linalg.LinAlgError:
            return None, None

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

    try:
        return _dlt(source[best_inliers], target[best_inliers]), best_inliers
    except np.linalg.LinAlgError:
        return None, None


def find_homography_lmeds(source, target, max_iterations=2000, seed=0):
    """The homography mapping `source` onto `target` by least median of squares.

    `cv2.findHomography( ..., cv2.LMEDS )`. Where RANSAC counts how many
    correspondences fall inside a threshold the caller chose, this minimises the
    **median** squared error and needs no threshold at all -- which is why a
    caller reaches for it when it cannot say what a good reprojection is. The
    price is that it tolerates at most half the correspondences being wrong,
    where RANSAC with a generous threshold tolerates more.

    The inlier mask comes from the robust scale of the winning model, the usual
    `1.4826 * (1 + 5 / (n - 4)) * sqrt( median )` with a 2.5 sigma cut, and the
    result is refit on it.

    Returns `(homography, mask)`, or `(None, None)` when there are fewer than
    four correspondences.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)

    if len(source) != len(target):
        raise ValueError("source and target must have the same length")
    if len(source) < 4:
        return None, None
    if len(source) == 4:
        try:
            return _dlt(source, target), np.ones(4, dtype=bool)
        except np.linalg.LinAlgError:
            return None, None

    rng = np.random.default_rng(seed)
    count = len(source)
    best_median = np.inf
    best = None
    # LMEDS assumes at least 50% inliers. A 99% chance of one all-inlier
    # four-point sample needs 72 draws, not an unconditional 2000.
    draws = min(max_iterations, int(np.ceil(np.log(0.01) / np.log(1 - 0.5 ** 4))))
    for start in range(0, draws, 32):
        samples = np.array([rng.choice(count, 4, replace=False)
                            for _ in range(min(32, draws - start))])
        src, dst = source[samples], target[samples]
        centres = [points.mean(axis=1, keepdims=True) for points in (src, dst)]
        shifted = [points - centre for points, centre in zip((src, dst), centres)]
        scales = [np.sqrt(2.) / np.maximum(np.linalg.norm(points, axis=2).mean(axis=1), 1e-12)
                  for points in shifted]
        src, dst = [points * scale[:, None, None] for points, scale in zip(shifted, scales)]
        design = np.zeros((len(samples), 8, 9))
        x, y = src[..., 0], src[..., 1]
        u, v = dst[..., 0], dst[..., 1]
        design[:, 0::2, 0:3] = np.stack((-x, -y, -np.ones_like(x)), axis=-1)
        design[:, 1::2, 3:6] = np.stack((-x, -y, -np.ones_like(x)), axis=-1)
        design[:, 0::2, 6:9] = np.stack((u*x, u*y, u), axis=-1)
        design[:, 1::2, 6:9] = np.stack((v*x, v*y, v), axis=-1)
        # NumPy executes the batch of small SVDs in native code.
        _, singular, vt = np.linalg.svd(design)
        valid = singular[:, 7] > singular[:, 0] * 1e-12
        transforms = []
        for centre, scale in zip(centres, scales):
            transform = np.broadcast_to(np.eye(3), (len(samples), 3, 3)).copy()
            transform[:, 0, 0] = transform[:, 1, 1] = scale
            transform[:, :2, 2] = -centre[:, 0] * scale[:, None]
            transforms.append(transform)
        candidates = np.linalg.inv(transforms[1]) @ vt[:, -1].reshape(-1, 3, 3) @ transforms[0]
        valid &= np.linalg.matrix_rank(candidates) == 3
        candidates = candidates[valid]
        if not len(candidates):
            continue
        points = np.column_stack((source, np.ones(count)))
        mapped = points @ candidates.transpose(0, 2, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = ((mapped[..., :2] / mapped[..., 2:] - target) ** 2).sum(axis=2)
        errors[~np.isfinite(errors)] = np.inf
        medians = np.median(errors, axis=1)
        winner = int(np.argmin(medians))
        if medians[winner] < best_median:
            best_median, best = medians[winner], candidates[winner]

    if best is None:
        return None, None
    scale = max(0.001, 2.5 * 1.4826 * (1.0 + 5.0 / max(1, count - 4)) * np.sqrt(best_median))
    squared = ((apply_homography(best, source) - target) ** 2).sum(axis=1)
    inliers = squared <= scale ** 2
    if int(inliers.sum()) < 4:
        return None, None
    try:
        return _dlt(source[inliers], target[inliers]), inliers
    except np.linalg.LinAlgError:
        return None, None


def _affine_from(source, target):
    """The least-squares affine taking `source` to `target`, as a 2 by 3.

    Three correspondences determine it exactly and more over-determine it; both
    are the same normal equations, so there is one path.
    """
    count = len(source)
    design = np.hstack([source, np.ones((count, 1))])
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3 or np.linalg.matrix_rank(solution[:2]) < 2:
        raise np.linalg.LinAlgError("degenerate affine correspondences")
    return solution.T


def estimate_affine_2d(source, target, threshold=3.0, confidence=0.99,
                       max_iterations=2000, seed=0, full=True):
    """The affine mapping `source` onto `target`, and its inlier mask.

    `cv2.estimateAffine2D`, and with `full` false `cv2.estimateAffinePartial2D`
    -- a similarity, four degrees of freedom rather than six. RANSAC over
    three-point samples (two for the partial form), refit on the consensus by
    least squares. `threshold` is the reprojection distance in pixels at which a
    correspondence counts, as `ransacReprojThreshold` is.

    Also what `cv2.estimateRigidTransform` was before OpenCV removed it in 4.x:
    `estimateRigidTransform( a, b, fullAffine )` is this with `full=fullAffine`,
    which is the migration OpenCV's own deprecation notice names. A caller still
    invoking it on a modern cv2 is calling a function that is not there.

    Returns `(matrix, mask)` with matrix 2 by 3, or `(None, None)` when there
    are too few correspondences or no consensus.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)

    if len(source) != len(target):
        raise ValueError("source and target must have the same length")

    needed = 3 if full else 2

    if len(source) < needed:
        return None, None

    def fit(chosen):
        if full:
            return _affine_from(source[chosen], target[chosen])

        # A similarity from two points: the complex number taking one
        # difference to the other is the rotation and the scale together.
        a, b = source[chosen[0]], source[chosen[1]]
        c, d = target[chosen[0]], target[chosen[1]]
        span = complex(*(b - a))

        if span == 0:
            raise np.linalg.LinAlgError("coincident points")

        factor = complex(*(d - c)) / span
        offset = complex(*c) - factor * complex(*a)

        return np.array([[factor.real, -factor.imag, offset.real],
                         [factor.imag, factor.real, offset.imag]])

    def residual(matrix):
        mapped = source @ matrix[:, :2].T + matrix[:, 2]

        return np.sqrt(((mapped - target) ** 2).sum(axis=1))

    if len(source) == needed:
        try:
            return fit(np.arange(needed)), np.ones(needed, dtype=bool)
        except np.linalg.LinAlgError:
            return None, None

    rng = np.random.default_rng(seed)
    count = len(source)
    best_inliers = np.zeros(count, dtype=bool)
    best_total = 0
    iterations = max_iterations
    step = 0

    while step < min(iterations, max_iterations):
        step += 1
        sample = rng.choice(count, needed, replace=False)

        try:
            candidate = fit(sample)
        except np.linalg.LinAlgError:
            continue

        if not np.all(np.isfinite(candidate)):
            continue

        inliers = residual(candidate) < threshold
        total = int(inliers.sum())

        if total > best_total:
            best_total, best_inliers = total, inliers
            ratio = total / count

            if ratio > 0:
                denominator = np.log(max(1e-12, 1.0 - ratio ** needed))
                iterations = int(
                    np.log(max(1e-12, 1.0 - confidence)) / denominator) + 1

    if best_total < needed:
        return None, None

    if full:
        try:
            return _affine_from(source[best_inliers], target[best_inliers]), best_inliers
        except np.linalg.LinAlgError:
            return None, None

    # A similarity refit on the consensus: the same normal equations in the
    # (a, b, tx, ty) parameterisation, where the matrix is [[a, -b], [b, a]].
    chosen = np.flatnonzero(best_inliers)
    x, y = source[chosen, 0], source[chosen, 1]
    zero = np.zeros(len(chosen))
    one = np.ones(len(chosen))
    design = np.vstack([np.column_stack([x, -y, one, zero]),
                        np.column_stack([y, x, zero, one])])
    observed = np.concatenate([target[chosen, 0], target[chosen, 1]])
    (a, b, tx, ty), _, _, _ = np.linalg.lstsq(design, observed, rcond=None)

    return np.array([[a, -b, tx], [b, a, ty]]), best_inliers
