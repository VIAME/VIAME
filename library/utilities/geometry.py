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
