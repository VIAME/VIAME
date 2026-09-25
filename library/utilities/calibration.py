# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Camera calibration from views of a planar target.

What `cv::calibrateCamera` and `cv::stereoCalibrate` did. Zhang's method
throughout: a homography per view, a closed form for the intrinsics from the
orthonormality of the rotation columns, extrinsics back out of each
homography, and then a bundle adjustment over the lot.

This is held to **accuracy**, not to OpenCV's digits. A calibration is the
solution of a non-convex least squares problem, so two implementations that
are both right land on different members of a flat minimum; the golden that
covers it compares against the rig the views were rendered through for
exactly that reason. Measured on that rig, both this and OpenCV recover the
focal lengths to about 1e-6 relative.

`flags` takes the `cv2.CALIB_*` names as strings, because the constants are
what is going away:

    fix_aspect_ratio      fy is tied to fx
    fix_principal_point   the centre is held
    zero_tangent_dist     p1 and p2 are held at zero
    fix_k1, fix_k2, fix_k3  that radial term is held at zero
    use_intrinsic_guess   hold them at what was *seeded*, not at the default
    fix_intrinsics        stereo only: solve the pose, leave K and d alone

**The fixing flags do not hold what you passed in.** Without
`use_intrinsic_guess` OpenCV holds each at its own default, discarding
whatever the seed said: the aspect ratio at exactly 1, the principal point
at the image centre `((width - 1) / 2, (height - 1) / 2)`, and every fixed
distortion coefficient at **zero**. Holding the seed instead costs half a
per cent on the principal point -- most of the tolerance the calibration
golden allows -- and leaves distortion on a rig that has none, which the
same golden checks for separately.
"""

import numpy as np


# The distortion vector is OpenCV's order: k1, k2, p1, p2, k3.
DISTORTION_TERMS = 5


def _least_squares():
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - a missing dependency
        raise ImportError(
            "camera calibration needs scipy.optimize.least_squares") from exc
    return least_squares


def _rodrigues(vector):
    """A rotation matrix from an axis-angle vector."""
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))

    if angle < 1e-12:
        return np.eye(3)

    axis = vector / angle
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])

    return (np.cos(angle) * np.eye(3) +
            np.sin(angle) * cross +
            (1.0 - np.cos(angle)) * np.outer(axis, axis))


def _inverse_rodrigues(matrix):
    """The axis-angle vector of a rotation matrix."""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)

    # Nearest true rotation first: the closed-form extrinsics below are not
    # exactly orthonormal, and the logarithm of a not-quite-rotation is not
    # quite a rotation vector.
    u, _, vt = np.linalg.svd(matrix)
    matrix = u @ vt
    if np.linalg.det(matrix) < 0:
        u[:, 2] *= -1
        matrix = u @ vt

    angle = np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))

    if angle < 1e-12:
        return np.zeros(3)

    if abs(angle - np.pi) < 1e-6:
        # Near pi the antisymmetric part vanishes; take the axis from the
        # symmetric part instead.
        symmetric = (matrix + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(symmetric), 0.0, None))
        if axis[0] > 1e-9:
            axis[1] = np.copysign(axis[1], symmetric[0, 1])
            axis[2] = np.copysign(axis[2], symmetric[0, 2])
        elif axis[1] > 1e-9:
            axis[2] = np.copysign(axis[2], symmetric[1, 2])
        return axis / np.linalg.norm(axis) * angle

    axis = np.array([matrix[2, 1] - matrix[1, 2],
                     matrix[0, 2] - matrix[2, 0],
                     matrix[1, 0] - matrix[0, 1]])
    return axis / (2.0 * np.sin(angle)) * angle


def _normalise(points):
    """Hartley conditioning: centroid at the origin, mean distance root two."""
    centre = points.mean(axis=0)
    shifted = points - centre
    scale = np.sqrt(2.0) / max(np.sqrt((shifted ** 2).sum(axis=1)).mean(), 1e-12)

    transform = np.array([[scale, 0.0, -scale * centre[0]],
                          [0.0, scale, -scale * centre[1]],
                          [0.0, 0.0, 1.0]])
    return shifted * scale, transform


def _view_homography(plane, image):
    """The homography taking the target's plane onto one view of it."""
    normalised_plane, tp = _normalise(plane)
    normalised_image, ti = _normalise(image)

    rows = []
    for (x, y), (u, v) in zip(normalised_plane, normalised_image):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    _, _, vt = np.linalg.svd(np.asarray(rows))
    homography = np.linalg.inv(ti) @ vt[-1].reshape(3, 3) @ tp

    return homography / homography[2, 2]


def _intrinsics_from_homographies(homographies):
    """Zhang's closed form for K, from the orthonormal rotation columns.

    Each view gives two constraints on `B = K^-T K^-1`: the first two columns
    of the homography are a scaled rotation, so they are orthogonal and of
    equal length. Six unknowns, so three views are the minimum -- with fewer
    the system is rank deficient and the caller gets a seeded guess instead.
    """
    def constraint(homography, i, j):
        a, b = homography[:, i], homography[:, j]
        return np.array([
            a[0] * b[0],
            a[0] * b[1] + a[1] * b[0],
            a[1] * b[1],
            a[2] * b[0] + a[0] * b[2],
            a[2] * b[1] + a[1] * b[2],
            a[2] * b[2]])

    rows = []
    for homography in homographies:
        rows.append(constraint(homography, 0, 1))
        rows.append(constraint(homography, 0, 0) -
                    constraint(homography, 1, 1))

    _, _, vt = np.linalg.svd(np.asarray(rows))
    b11, b12, b22, b13, b23, b33 = vt[-1]

    denominator = b11 * b22 - b12 * b12
    if abs(denominator) < 1e-20:
        return None

    cy = (b12 * b13 - b11 * b23) / denominator
    lam = b33 - (b13 * b13 + cy * (b12 * b13 - b11 * b23)) / b11

    if lam / b11 <= 0 or lam <= 0:
        return None

    fx = np.sqrt(lam / b11)
    fy = np.sqrt(lam * b11 / denominator)
    skew = -b12 * fx * fx * fy / lam
    cx = skew * cy / fy - b13 * fx * fx / lam

    if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        return None

    # The skew is dropped: `cv::calibrateCamera` fixes it at zero and every
    # VIAME caller assumes a zero-skew matrix.
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _extrinsics_from_homography(homography, intrinsics):
    """The view's pose, from its homography and the shared intrinsics."""
    inverse = np.linalg.inv(intrinsics)
    columns = inverse @ homography

    scale = 1.0 / max(np.linalg.norm(columns[:, 0]), 1e-12)
    r1 = columns[:, 0] * scale
    r2 = columns[:, 1] * scale
    translation = columns[:, 2] * scale

    # A target seen from behind the camera is the mirrored solution
    if translation[2] < 0:
        r1, r2, translation = -r1, -r2, -translation

    r3 = np.cross(r1, r2)
    rotation = np.column_stack([r1, r2, r3])

    return _inverse_rodrigues(rotation), translation


def project_points(points, rotation, translation, intrinsics, distortion):
    """Brown-Conrady projection, vectorised over the points.

    The same model `projection.project_points` implements in C++; it is
    repeated here because the optimiser calls it a few hundred thousand
    times and wants the whole array at once.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera = points @ _rodrigues(rotation).T + np.asarray(translation).reshape(3)

    z = np.where(np.abs(camera[:, 2]) < 1e-12, 1e-12, camera[:, 2])
    x = camera[:, 0] / z
    y = camera[:, 1] / z

    k1, k2, p1, p2, k3 = distortion
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2

    distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    distorted_y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    return np.column_stack([
        intrinsics[0, 0] * distorted_x + intrinsics[0, 2],
        intrinsics[1, 1] * distorted_y + intrinsics[1, 2]])


def _seed_intrinsics(image_size):
    """A usable starting point when the closed form will not produce one."""
    width, height = image_size
    focal = float(max(width, height))
    return np.array([[focal, 0.0, (width - 1) / 2.0],
                     [0.0, focal, (height - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def _pack(intrinsics, distortion, poses, flags):
    """The free parameters, in the order `_unpack` reads them back."""
    values = []

    if "fix_aspect_ratio" in flags:
        values.append(intrinsics[0, 0])
    else:
        values.extend([intrinsics[0, 0], intrinsics[1, 1]])

    if "fix_principal_point" not in flags:
        values.extend([intrinsics[0, 2], intrinsics[1, 2]])

    for index, name in enumerate(("fix_k1", "fix_k2", None, None, "fix_k3")):
        if index in (2, 3):
            if "zero_tangent_dist" not in flags:
                values.append(distortion[index])
        elif name not in flags:
            values.append(distortion[index])

    for rotation, translation in poses:
        values.extend(rotation)
        values.extend(translation)

    return np.asarray(values, dtype=np.float64)


def _unpack(values, intrinsics, distortion, count, flags):
    """`_pack`'s inverse: the parameters back into a calibration."""
    intrinsics = intrinsics.copy()
    distortion = np.asarray(distortion, dtype=np.float64).copy()
    at = 0

    if "fix_aspect_ratio" in flags:
        ratio = intrinsics[1, 1] / intrinsics[0, 0]
        intrinsics[0, 0] = values[at]
        intrinsics[1, 1] = values[at] * ratio
        at += 1
    else:
        intrinsics[0, 0], intrinsics[1, 1] = values[at], values[at + 1]
        at += 2

    if "fix_principal_point" not in flags:
        intrinsics[0, 2], intrinsics[1, 2] = values[at], values[at + 1]
        at += 2

    for index, name in enumerate(("fix_k1", "fix_k2", None, None, "fix_k3")):
        if index in (2, 3):
            if "zero_tangent_dist" not in flags:
                distortion[index] = values[at]
                at += 1
        elif name not in flags:
            distortion[index] = values[at]
            at += 1

    poses = []
    for _ in range(count):
        poses.append((values[at:at + 3], values[at + 3:at + 6]))
        at += 6

    return intrinsics, distortion, poses


def initial_camera_matrix(object_points, image_points, image_size,
                          aspect_ratio=0.0):
    """A seeded intrinsic matrix, which is `cv2.initCameraMatrix2D`.

    Zhang's closed form on its own, with no refinement. A caller that is
    about to run the full fit does not need this -- `calibrate_camera` seeds
    itself the same way -- but the progressive calibration wants the seed
    separately so it can watch the aspect ratio and principal point before
    deciding which of them to fix.

    `aspect_ratio` above zero ties fy to fx at that ratio, as OpenCV's does.
    """
    object_points = [np.asarray(p, dtype=np.float64).reshape(-1, 3)
                     for p in object_points]
    image_points = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
                    for p in image_points]

    homographies = [_view_homography(plane[:, :2], seen)
                    for plane, seen in zip(object_points, image_points)]

    intrinsics = None
    if len(homographies) >= 3:
        intrinsics = _intrinsics_from_homographies(homographies)
    if intrinsics is None:
        intrinsics = _seed_intrinsics(image_size)

    if aspect_ratio > 0:
        focal = (intrinsics[0, 0] + intrinsics[1, 1] * aspect_ratio) / 2.0
        intrinsics[0, 0] = focal
        intrinsics[1, 1] = focal / aspect_ratio

    return intrinsics


def calibrate_camera(object_points, image_points, image_size, flags=(),
                     intrinsics=None, distortion=None):
    """Intrinsics, distortion and a pose per view, from a planar target.

    `cv2.calibrateCamera`. `object_points` and `image_points` are one array
    per view -- N by 3 and N by 2 -- and the target must be planar, which is
    what every VIAME caller has.

    `intrinsics` and `distortion` seed the fit when given, which is how the
    progressive calibration hands one round's answer to the next; the closed
    form is used when they are not.

    Returns `(rms, intrinsics, distortion, rotations, translations)`, where
    `rms` is the root-mean-square reprojection error in pixels and the
    rotations are axis-angle, both as `cv2.calibrateCamera` returns them.
    """
    flags = set(flags)
    object_points = [np.asarray(p, dtype=np.float64).reshape(-1, 3)
                     for p in object_points]
    image_points = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
                    for p in image_points]

    if len(object_points) != len(image_points):
        raise ValueError("one set of image points per view of the target")
    if not object_points:
        raise ValueError("calibration needs at least one view")

    for plane in object_points:
        if np.abs(plane[:, 2] - plane[0, 2]).max() > 1e-9:
            raise ValueError(
                "calibrate_camera wants a planar target; the object points "
                "are not coplanar")

    homographies = []
    for plane, seen in zip(object_points, image_points):
        homographies.append(_view_homography(plane[:, :2], seen))

    if intrinsics is None:
        if len(homographies) >= 3:
            intrinsics = _intrinsics_from_homographies(homographies)
        if intrinsics is None:
            intrinsics = _seed_intrinsics(image_size)

    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3).copy()

    # The defaults the two fixing flags snap to, unless the caller says the
    # seed is meant. See the module docstring: this is not what the flag
    # names suggest and it is what OpenCV does.
    if "use_intrinsic_guess" not in flags:
        if "fix_aspect_ratio" in flags:
            intrinsics[1, 1] = intrinsics[0, 0]
        if "fix_principal_point" in flags:
            intrinsics[0, 2] = (image_size[0] - 1) / 2.0
            intrinsics[1, 2] = (image_size[1] - 1) / 2.0

    if distortion is None:
        distortion = np.zeros(DISTORTION_TERMS)
    else:
        distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
        distortion = np.pad(distortion[:DISTORTION_TERMS],
                            (0, max(0, DISTORTION_TERMS - len(distortion))))

    if "use_intrinsic_guess" not in flags:
        for index, name in enumerate(
                ("fix_k1", "fix_k2", None, None, "fix_k3")):
            if index in (2, 3):
                if "zero_tangent_dist" in flags:
                    distortion[index] = 0.0
            elif name in flags:
                distortion[index] = 0.0

    poses = [_extrinsics_from_homography(h, intrinsics) for h in homographies]

    def residuals(values):
        k, d, current = _unpack(values, intrinsics, distortion,
                                len(object_points), flags)
        out = []
        for (rotation, translation), plane, seen in zip(current, object_points,
                                                        image_points):
            out.append((project_points(plane, rotation, translation, k, d) -
                        seen).ravel())
        return np.concatenate(out)

    solution = _least_squares()(
        residuals, _pack(intrinsics, distortion, poses, flags),
        method="lm", max_nfev=2000)

    intrinsics, distortion, poses = _unpack(
        solution.x, intrinsics, distortion, len(object_points), flags)

    total = float(np.sum(solution.fun ** 2))
    rms = float(np.sqrt(total / (len(solution.fun) / 2.0)))

    rotations = [np.asarray(r, dtype=np.float64) for r, _ in poses]
    translations = [np.asarray(t, dtype=np.float64) for _, t in poses]

    return rms, intrinsics, distortion, rotations, translations


def stereo_calibrate(object_points, left_points, right_points,
                     left_intrinsics, left_distortion,
                     right_intrinsics, right_distortion,
                     image_size, flags=("fix_intrinsics",)):
    """The pose of the right camera in the left camera's frame.

    `cv2.stereoCalibrate`. Only `fix_intrinsics` is supported, because it is
    the only way VIAME calls it: both cameras are calibrated on their own
    first and this solves the rig geometry over the pair.

    Returns `(rms, rotation, translation, essential, fundamental)`, with the
    rotation as a three by three.
    """
    flags = set(flags)

    if "fix_intrinsics" not in flags:
        raise ValueError(
            "stereo_calibrate only implements fix_intrinsics; calibrate each "
            "camera first")

    object_points = [np.asarray(p, dtype=np.float64).reshape(-1, 3)
                     for p in object_points]
    left_points = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
                   for p in left_points]
    right_points = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
                    for p in right_points]

    left_intrinsics = np.asarray(left_intrinsics, dtype=np.float64)
    right_intrinsics = np.asarray(right_intrinsics, dtype=np.float64)
    left_distortion = np.asarray(left_distortion,
                                 dtype=np.float64).reshape(-1)[:DISTORTION_TERMS]
    right_distortion = np.asarray(right_distortion,
                                  dtype=np.float64).reshape(-1)[:DISTORTION_TERMS]
    left_distortion = np.pad(left_distortion,
                             (0, DISTORTION_TERMS - len(left_distortion)))
    right_distortion = np.pad(right_distortion,
                              (0, DISTORTION_TERMS - len(right_distortion)))

    # Each view's pose in each camera, then the relative pose they imply.
    # Averaging the rotations through their axis-angle vectors would be wrong
    # near a half turn; averaging the matrices and re-orthonormalising is not.
    left_poses, right_poses, relatives = [], [], []

    for plane, seen_left, seen_right in zip(object_points, left_points,
                                            right_points):
        h_left = _view_homography(plane[:, :2], seen_left)
        h_right = _view_homography(plane[:, :2], seen_right)

        rl, tl = _extrinsics_from_homography(h_left, left_intrinsics)
        rr, tr = _extrinsics_from_homography(h_right, right_intrinsics)

        left_poses.append((rl, tl))
        right_poses.append((rr, tr))

        rotation_left = _rodrigues(rl)
        rotation_right = _rodrigues(rr)
        relative = rotation_right @ rotation_left.T
        relatives.append((relative, tr - relative @ tl))

    mean_rotation = sum(r for r, _ in relatives) / len(relatives)
    u, _, vt = np.linalg.svd(mean_rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, 2] *= -1
        rotation = u @ vt

    translation = sum(t for _, t in relatives) / len(relatives)

    # Refine the rig pose and every view's left pose together; the right pose
    # of a view is the left one composed with the rig, which is the constraint
    # that makes this a stereo calibration rather than two separate ones.
    def pack(rig_rotation, rig_translation, poses):
        values = list(_inverse_rodrigues(rig_rotation)) + list(rig_translation)
        for r, t in poses:
            values.extend(r)
            values.extend(t)
        return np.asarray(values, dtype=np.float64)

    def unpack(values):
        rig = (_rodrigues(values[0:3]), values[3:6])
        poses, at = [], 6
        for _ in range(len(object_points)):
            poses.append((values[at:at + 3], values[at + 3:at + 6]))
            at += 6
        return rig, poses

    def residuals(values):
        (rig_rotation, rig_translation), poses = unpack(values)
        out = []
        for (rotation_vector, translation), plane, seen_left, seen_right in zip(
                poses, object_points, left_points, right_points):
            out.append((project_points(plane, rotation_vector, translation,
                                       left_intrinsics, left_distortion) -
                        seen_left).ravel())

            composed = rig_rotation @ _rodrigues(rotation_vector)
            shifted = rig_rotation @ np.asarray(translation) + rig_translation
            out.append((project_points(plane,
                                       _inverse_rodrigues(composed), shifted,
                                       right_intrinsics, right_distortion) -
                        seen_right).ravel())
        return np.concatenate(out)

    solution = _least_squares()(
        residuals, pack(rotation, translation, left_poses),
        method="lm", max_nfev=2000)

    (rotation, translation), _ = unpack(solution.x)
    translation = np.asarray(translation, dtype=np.float64)

    rms = float(np.sqrt(np.sum(solution.fun ** 2) /
                        (len(solution.fun) / 2.0)))

    cross = np.array([[0.0, -translation[2], translation[1]],
                      [translation[2], 0.0, -translation[0]],
                      [-translation[1], translation[0], 0.0]])
    essential = cross @ rotation
    fundamental = (np.linalg.inv(right_intrinsics).T @ essential @
                   np.linalg.inv(left_intrinsics))

    norm = np.abs(fundamental).max()
    if norm > 1e-12:
        fundamental = fundamental / norm

    return rms, rotation, translation, essential, fundamental
