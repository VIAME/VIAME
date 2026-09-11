"""Drive the feature chain over the golden fixtures.

Shared by the recorder and the golden test, like `runner.py`, so that both
exercise the implementations the same way. Nothing here knows which
implementation is OpenCV's and which is the replacement.

The values are arrays rather than images, so they go to `.npz` with named
members instead of through `imageio_utils`.
"""

import numpy as np

import feature_cases
import runner


# A vital feature as the five numbers `vital::feature` actually carries.
# `color` and `covariance` are never set by any of these implementations.
FEATURE_COLUMNS = ("x", "y", "scale", "angle", "magnitude")


def save(path, **arrays):
    """Write named arrays to `path` + '.npz'; return the file written."""
    target = str(path) + ".npz"
    np.savez_compressed(target, **arrays)
    return target


def load(path):
    """Read back a file written by `save`, as a plain dict."""
    with np.load(str(path)) as data:
        return {name: data[name] for name in data.files}


def features_to_array(feature_set):
    """The feature set as an (n, 5) array in `FEATURE_COLUMNS` order."""
    rows = []

    for feature in feature_set.features():
        location = np.asarray(feature.location, dtype=np.float64)
        rows.append([location[0], location[1], feature.scale,
                     feature.angle, feature.magnitude])

    return np.array(rows, dtype=np.float64).reshape(len(rows), 5)


def descriptors_to_array(descriptor_set):
    """The descriptor set as an (n, d) array.

    `todoublearray` rather than `tobytearray`: SIFT and SURF are float
    descriptors, and the byte view would only be a reinterpretation.
    """
    rows = [np.asarray(d.todoublearray(), dtype=np.float64)
            for d in descriptor_set.descriptors()]

    if not rows:
        return np.zeros((0, 0), dtype=np.float64)

    return np.vstack(rows)


# ----------------------------------------------------------------------------
# Comparing a case whose implementation is not deterministic
# ----------------------------------------------------------------------------

def compare_unstable(kind, member, got, want):  # noqa: C901
    """Problems with an unstable case's member, as a list of strings.

    `ocv_flann_based` builds randomised KD-trees, so a match set and anything
    downstream of it differ run to run -- see `feature_cases.UNSTABLE`. What
    is still a contract:

    * a match set keeps its size to within `MATCH_COUNT_TOLERANCE` and still
      contains `MATCH_AGREEMENT` of the pairs the recording has;
    * a track set's feature locations are exact, because those are the
      detector's and the detector is deterministic, while how many of them
      link into a two-frame track is the matcher's and may move by
      `TRACK_LENGTH_TOLERANCE`.

    The locations being exact is not a weak check dressed up: they are what a
    port that lost the extractor's replaced feature set would leave untouched
    while making the linking nonsense, so the two halves together say more
    than either alone.
    """
    if kind == "matches" and member == "matches":
        return _compare_pairs(got, want)

    if kind == "tracks" and member == "states":
        return _compare_states(got, want)

    return ["no rule for comparing an unstable '{}' member '{}'".format(
        kind, member)]


def _compare_pairs(got, want):
    problems = []

    if len(want):
        drift = abs(len(got) - len(want)) / float(len(want))
        if drift > feature_cases.MATCH_COUNT_TOLERANCE:
            problems.append(
                "{} matches, recorded {}: {:.1%} apart, more than {:.0%}"
                .format(len(got), len(want), drift,
                        feature_cases.MATCH_COUNT_TOLERANCE))

    recorded = {tuple(row) for row in want}
    actual = {tuple(row) for row in got}

    if recorded:
        agreement = len(recorded & actual) / float(len(recorded))
        if agreement < feature_cases.MATCH_AGREEMENT:
            problems.append(
                "{:.1%} of the recorded pairs are still matched, less than "
                "{:.0%}".format(agreement, feature_cases.MATCH_AGREEMENT))

    return problems


def _compare_states(got, want):
    problems = []

    # (frame, x, y): the detector's, and exact
    recorded = {(row[0], row[2], row[3]) for row in want}
    actual = {(row[0], row[2], row[3]) for row in got}

    missing = recorded - actual
    if missing:
        problems.append(
            "{} of {} recorded feature locations are gone".format(
                len(missing), len(recorded)))

    extra = actual - recorded
    if extra:
        problems.append(
            "{} feature locations the recording does not have".format(
                len(extra)))

    # How many tracks have each length: the matcher's, and approximate
    for length, count in sorted(_length_counts(want).items()):
        found = _length_counts(got).get(length, 0)
        drift = abs(found - count) / float(count)

        if drift > feature_cases.TRACK_LENGTH_TOLERANCE:
            problems.append(
                "{} tracks of length {}, recorded {}: {:.1%} apart, more "
                "than {:.0%}".format(found, length, count, drift,
                                     feature_cases.TRACK_LENGTH_TOLERANCE))

    return problems


def _length_counts(states):
    """How many tracks have each number of states."""
    lengths = {}
    for row in states:
        lengths[row[1]] = lengths.get(row[1], 0) + 1

    counts = {}
    for length in lengths.values():
        counts[length] = counts.get(length, 0) + 1

    return counts


def _container(array):
    from kwiver.vital.types import Image, ImageContainer
    return ImageContainer(Image(np.ascontiguousarray(array)))


def detect_and_extract(impl, config, array):
    """Detect features and extract their descriptors, as the chain does.

    One call for both because that is how `track_features:core` uses them:
    the feature set the detector returns is handed straight to the extractor
    of the same name, which is free to recognise its own kind and read what
    `vital::feature` cannot carry.
    """
    from kwiver.vital.algo import DetectFeatures, ExtractDescriptors

    detector = DetectFeatures.create(impl)
    if detector is None:
        raise RuntimeError("detect_features '{}' is not registered".format(impl))

    extractor = ExtractDescriptors.create(impl)
    if extractor is None:
        raise RuntimeError(
            "extract_descriptors '{}' is not registered".format(impl))

    runner._configure(detector, config)
    runner._configure(extractor, config)

    image = _container(array)
    features = detector.detect(image)
    detected = features_to_array(features)

    descriptors, features = extractor.extract(image, features)

    return {
        "detected": detected,
        "features": features_to_array(features),
        "descriptors": descriptors_to_array(descriptors),
    }


def _feature_pair(impl, config, arrays):
    """Detect and extract on each of two images, keeping the vital objects."""
    from kwiver.vital.algo import DetectFeatures, ExtractDescriptors

    detector = DetectFeatures.create(impl)
    extractor = ExtractDescriptors.create(impl)
    runner._configure(detector, config)
    runner._configure(extractor, config)

    out = []
    for array in arrays:
        image = _container(array)
        features = detector.detect(image)
        descriptors, features = extractor.extract(image, features)
        out.append((features, descriptors))

    return out


def match(impl, config, feature_impl, feature_config, arrays):
    """Match the descriptors of two images, as index pairs."""
    from kwiver.vital.algo import MatchFeatures

    matcher = MatchFeatures.create(impl)
    if matcher is None:
        raise RuntimeError("match_features '{}' is not registered".format(impl))

    runner._configure(matcher, config)

    (feat1, desc1), (feat2, desc2) = _feature_pair(
        feature_impl, feature_config, arrays)

    matches = matcher.match(feat1, desc1, feat2, desc2)
    pairs = list(matches.matches()) if matches else []

    return {
        "matches": np.array(pairs, dtype=np.int64).reshape(len(pairs), 2),
    }


def track(impl, config, arrays):
    """Track features across the images, as `stabilize_image` does.

    Recorded as one row per track state: (frame, track id, x, y). That is
    what a tracker is for, and it is also where a mis-paired descriptor
    shows up -- a feature matched to the wrong one lands in the wrong track.
    """
    from kwiver.vital.algo import TrackFeatures

    tracker = TrackFeatures.create(impl)
    if tracker is None:
        raise RuntimeError("track_features '{}' is not registered".format(impl))

    runner._configure(tracker, config)

    tracks = None
    for frame, array in enumerate(arrays):
        tracks = tracker.track(tracks, frame, _container(array), None)

    rows = []
    for track in (tracks.tracks() if tracks else []):
        for state in track:
            feature = state.feature
            location = np.asarray(feature.location, dtype=np.float64)
            rows.append([state.frame_id, track.id, location[0], location[1]])

    rows.sort()

    return {
        "states": np.array(rows, dtype=np.float64).reshape(len(rows), 4),
    }


def homography_correspondences():
    """A planar scene mapped by a known homography, with outliers.

    Deterministic, so the RANSAC estimator's answer is too; see
    `feature_cases` for why the matched features are not used here.
    """
    state = np.random.RandomState(feature_cases.SYNTHETIC_SEED)
    count = feature_cases.HOMOGRAPHY_POINTS

    source = state.uniform(20.0, 460.0, size=(count, 2))
    matrix = np.array(feature_cases.HOMOGRAPHY_TRUE, dtype=np.float64)

    homogeneous = np.hstack([source, np.ones((count, 1))]) @ matrix.T
    target = homogeneous[:, :2] / homogeneous[:, 2:]

    stride = feature_cases.HOMOGRAPHY_OUTLIER_STRIDE
    spread = feature_cases.HOMOGRAPHY_OUTLIER_SPREAD
    outliers = target[::stride].shape[0]
    target[::stride] += state.uniform(-spread, spread, size=(outliers, 2))

    return ([np.array(point) for point in source],
            [np.array(point) for point in target])


def fundamental_correspondences():
    """A box of 3D points seen by two cameras, with outliers.

    Non-planar on purpose: a fundamental matrix is degenerate on a plane, so
    the homography fixture would not exercise this estimator at all.
    """
    state = np.random.RandomState(feature_cases.SYNTHETIC_SEED)
    count = feature_cases.FUNDAMENTAL_POINTS

    scene = state.uniform(-1.0, 1.0, size=(count, 3))
    scene[:, 2] += feature_cases.FUNDAMENTAL_DEPTH

    focal = feature_cases.FUNDAMENTAL_FOCAL
    cx, cy = feature_cases.FUNDAMENTAL_CENTRE
    intrinsics = np.array([[focal, 0.0, cx],
                           [0.0, focal, cy],
                           [0.0, 0.0, 1.0]])

    rotation = _rodrigues(np.array(feature_cases.FUNDAMENTAL_ROTATION))
    translation = np.array(feature_cases.FUNDAMENTAL_TRANSLATION).reshape(3, 1)

    first = _project(intrinsics, scene)
    second = _project(intrinsics, (rotation @ scene.T + translation).T)

    stride = feature_cases.FUNDAMENTAL_OUTLIER_STRIDE
    spread = feature_cases.FUNDAMENTAL_OUTLIER_SPREAD
    outliers = second[::stride].shape[0]
    second[::stride] += state.uniform(-spread, spread, size=(outliers, 2))

    return ([np.array(point) for point in first],
            [np.array(point) for point in second])


def _rodrigues(vector):
    """A rotation matrix from an axis-angle vector, without cv2."""
    angle = float(np.linalg.norm(vector))

    if angle == 0.0:
        return np.eye(3)

    axis = vector / angle
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])

    return (np.eye(3) + np.sin(angle) * cross +
            (1.0 - np.cos(angle)) * (cross @ cross))


def _project(intrinsics, points):
    projected = (intrinsics @ points.T).T
    return projected[:, :2] / projected[:, 2:]


def _estimate(algorithm, points1, points2, inlier_scale):
    """Call the estimator and describe what came back.

    `estimate` returns `(matrix, inliers)` -- see
    `python/kwiver/vital/algo/estimator_extras.cxx`, which is what makes the
    inlier flags visible from python at all. They are the half of the answer
    every C++ caller actually uses, so they are recorded.
    """
    matrix, inliers = algorithm.estimate(points1, points2, inlier_scale)

    return {
        "matrix": (np.zeros((0, 0)) if matrix is None
                   else np.asarray(matrix.matrix(), dtype=np.float64)),
        "inliers": np.array(list(inliers), dtype=np.int64),
    }


def estimate_homography(impl, config, inlier_scale):
    """The homography and inlier flags for the synthetic correspondences."""
    from kwiver.vital.algo import EstimateHomography

    algorithm = EstimateHomography.create(impl)
    if algorithm is None:
        raise RuntimeError(
            "estimate_homography '{}' is not registered".format(impl))

    runner._configure(algorithm, config)
    points1, points2 = homography_correspondences()

    return _estimate(algorithm, points1, points2, inlier_scale)


def estimate_fundamental(impl, config, inlier_scale):
    """The fundamental matrix and inlier flags, likewise."""
    from kwiver.vital.algo import EstimateFundamentalMatrix

    algorithm = EstimateFundamentalMatrix.create(impl)
    if algorithm is None:
        raise RuntimeError(
            "estimate_fundamental_matrix '{}' is not registered".format(impl))

    runner._configure(algorithm, config)
    points1, points2 = fundamental_correspondences()

    return _estimate(algorithm, points1, points2, inlier_scale)
