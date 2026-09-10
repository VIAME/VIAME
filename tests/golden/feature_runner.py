"""Drive the feature chain over the golden fixtures.

Shared by the recorder and the golden test, like `runner.py`, so that both
exercise the implementations the same way. Nothing here knows which
implementation is OpenCV's and which is the replacement.

The values are arrays rather than images, so they go to `.npz` with named
members instead of through `imageio_utils`.
"""

import numpy as np

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


def matched_points(feature_impl, feature_config, arrays):
    """The two point lists a matcher's output picks out, for the estimators."""
    from kwiver.vital.algo import MatchFeatures

    matcher = MatchFeatures.create("ocv_flann_based")

    (feat1, desc1), (feat2, desc2) = _feature_pair(
        feature_impl, feature_config, arrays)

    matches = matcher.match(feat1, desc1, feat2, desc2)
    pairs = list(matches.matches()) if matches else []

    features1 = feat1.features()
    features2 = feat2.features()

    points1 = [np.asarray(features1[i].location, dtype=np.float64)
               for i, _ in pairs]
    points2 = [np.asarray(features2[j].location, dtype=np.float64)
               for _, j in pairs]

    return points1, points2


def estimate_homography(impl, config, feature_impl, arrays, inlier_scale):
    """The 3x3 homography the estimator returns for the matched points."""
    from kwiver.vital.algo import EstimateHomography

    algorithm = EstimateHomography.create(impl)
    if algorithm is None:
        raise RuntimeError(
            "estimate_homography '{}' is not registered".format(impl))

    runner._configure(algorithm, config)

    points1, points2 = matched_points(feature_impl, {}, arrays)

    inliers = []
    homography = algorithm.estimate(points1, points2, inliers, inlier_scale)

    return {
        "matrix": (np.zeros((0, 0)) if homography is None
                   else np.asarray(homography.matrix(), dtype=np.float64)),
        "points": np.array(len(points1), dtype=np.int64),
    }


def estimate_fundamental(impl, config, feature_impl, arrays, inlier_scale):
    """The 3x3 fundamental matrix the estimator returns."""
    from kwiver.vital.algo import EstimateFundamentalMatrix

    algorithm = EstimateFundamentalMatrix.create(impl)
    if algorithm is None:
        raise RuntimeError(
            "estimate_fundamental_matrix '{}' is not registered".format(impl))

    runner._configure(algorithm, config)

    points1, points2 = matched_points(feature_impl, {}, arrays)

    inliers = []
    matrix = algorithm.estimate(points1, points2, inliers, inlier_scale)

    return {
        "matrix": (np.zeros((0, 0)) if matrix is None
                   else np.asarray(matrix.matrix(), dtype=np.float64)),
        "points": np.array(len(points1), dtype=np.int64),
    }
