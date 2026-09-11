"""What the python feature chain guarantees beyond the golden replay.

`tests/golden/test_golden.py` holds `ocv_SIFT`, `ocv_SURF`,
`ocv_flann_based` and the two estimators to a recording of the C++ they
replaced. What is here is the part of the contract the recording cannot
reach, and the one place the port deliberately departs from it.

`tests/baseline/registry.json` is the compatibility contract, and
`compare_registry.py` cannot enforce it for a python implementation -- the
registry dump cannot introspect one's config -- so the keys and their
defaults are held here instead, as P4-T05's video readers are.

Run just these:  ctest -R "unit:image_processing"
"""

import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.abspath(os.path.join(HERE, "..", "..", "golden"))
REGISTRY = os.path.abspath(
    os.path.join(HERE, "..", "..", "baseline", "registry.json"))
FRAME = os.path.join(GOLDEN, "inputs", "frame_00.png")


@pytest.fixture(scope="module", autouse=True)
def modules():
    from kwiver.vital.modules import load_known_modules
    load_known_modules()


def recorded_config(interface, name):
    with open(REGISTRY) as handle:
        entry = json.load(handle)["algorithms"][interface][name]

    return {key: item["default"] for key, item in entry["config"].items()}


def create(interface, name):
    import kwiver.vital.algo as algo

    algorithm = getattr(algo, interface).create(name)
    assert algorithm is not None, \
        "{} '{}' is not registered".format(interface, name)
    return algorithm


def configured(interface, name, **values):
    from kwiver.vital.config import empty_config

    algorithm = create(interface, name)
    cfg = empty_config()
    for key, value in values.items():
        cfg.set_value(key, str(value))
    algorithm.set_configuration(cfg)
    return algorithm


def frame():
    import sys
    sys.path.insert(0, GOLDEN)
    import imageio_utils

    from kwiver.vital.types import Image, ImageContainer
    array = imageio_utils.load(FRAME)
    return ImageContainer(Image(np.ascontiguousarray(array)))


# The interface each name answers to, and the snake_case key `registry.json`
# files it under.
NAMES = [
    ("DetectFeatures", "detect_features", "ocv_SIFT"),
    ("DetectFeatures", "detect_features", "ocv_SURF"),
    ("ExtractDescriptors", "extract_descriptors", "ocv_SIFT"),
    ("ExtractDescriptors", "extract_descriptors", "ocv_SURF"),
    ("MatchFeatures", "match_features", "ocv_flann_based"),
    ("EstimateHomography", "estimate_homography", "ocv"),
    ("EstimateFundamentalMatrix", "estimate_fundamental_matrix", "ocv"),
]


@pytest.mark.parametrize("interface,key,name", NAMES,
                         ids=["{}:{}".format(k, n) for _, k, n in NAMES])
def test_names_keep_their_config(interface, key, name):
    cfg = create(interface, name).get_configuration()
    have = {k: cfg.get_value(k) for k in cfg.available_values()}

    for k, default in sorted(recorded_config(key, name).items()):
        assert k in have, "'{}' lost config key '{}'".format(name, k)
        assert have[k] == default, (
            "'{}' config key '{}' defaults to '{}', recorded '{}'".format(
                name, k, have[k], default))


# ----------------------------------------------------------------------------
# The divergence: the C++ wrappers ignored their configuration
# ----------------------------------------------------------------------------
#
# `cv::Ptr::constCast` returns a new Ptr by value, so
# `detector.constCast<...>() = create( ... )` in all four C++ wrappers
# assigned the reconfigured detector to a temporary and threw it away. Every
# call used the detector built at construction from the defaults, which is
# why every non-default variant in the recording is byte-identical to
# `defaults`. These are what say the port does not.

def test_sift_n_features_is_applied():
    few = configured("DetectFeatures", "ocv_SIFT", n_features=20)
    assert few.detect(frame()).size() == 20

    default = create("DetectFeatures", "ocv_SIFT")
    assert default.detect(frame()).size() > 20


def test_sift_contrast_threshold_is_applied():
    strict = configured("DetectFeatures", "ocv_SIFT", contrast_threshold=0.08)
    default = create("DetectFeatures", "ocv_SIFT")

    assert strict.detect(frame()).size() < default.detect(frame()).size()


def test_surf_hessian_threshold_is_applied():
    strict = configured("DetectFeatures", "ocv_SURF", hessian_threshold=500)
    default = create("DetectFeatures", "ocv_SURF")

    assert strict.detect(frame()).size() < default.detect(frame()).size()


def test_surf_extended_widens_the_descriptor():
    image = frame()

    for extended, width in (("false", 64), ("true", 128)):
        detector = configured("DetectFeatures", "ocv_SURF",
                              extended=extended, hessian_threshold=500)
        extractor = configured("ExtractDescriptors", "ocv_SURF",
                               extended=extended, hessian_threshold=500)

        descriptors, _ = extractor.extract(image, detector.detect(image))
        assert descriptors.descriptors()[0].size == width


# ----------------------------------------------------------------------------
# What the recording cannot reach
# ----------------------------------------------------------------------------

def test_the_extractor_reads_the_octave_the_detector_found():
    """A SIFT descriptor depends on the keypoint's octave.

    `vital::feature` cannot carry one, so the C++ bridge kept the keypoints
    inside its own feature set and the extractor recognised it. This is the
    python form of that: descriptors extracted from the detector's own
    feature set must differ from ones extracted after the octave has been
    thrown away, or the carrying is not happening.
    """
    from kwiver.vital.types import SimpleFeatureSet

    image = frame()
    detector = configured("DetectFeatures", "ocv_SIFT", n_features=20)
    extractor = configured("ExtractDescriptors", "ocv_SIFT", n_features=20)

    carried = detector.detect(image)
    stripped = SimpleFeatureSet(list(carried.features()))

    with_octave, _ = extractor.extract(image, carried)
    without, _ = extractor.extract(image, stripped)

    assert with_octave.size() == without.size()

    first = np.asarray(with_octave.descriptors()[0].todoublearray())
    second = np.asarray(without.descriptors()[0].todoublearray())
    assert not np.array_equal(first, second), (
        "the octave is not reaching the extractor: a feature set stripped of "
        "it gave the same descriptors")


def test_the_extractor_returns_the_feature_set_it_used():
    """`extract` returns `(descriptors, features)`, and the second matters.

    cv2 may drop a keypoint it cannot describe, and a caller left holding the
    set it passed in pairs every descriptor after the drop with the wrong
    feature.
    """
    image = frame()
    detector = create("DetectFeatures", "ocv_SIFT")
    extractor = create("ExtractDescriptors", "ocv_SIFT")

    descriptors, features = extractor.extract(image, detector.detect(image))
    assert features.size() == descriptors.size()


def test_the_estimators_return_their_inliers():
    """The half of the answer every C++ caller uses.

    `std::vector<bool>& inliers` is an output parameter, which neither the
    generated binding nor the generated trampoline could carry; both sides
    are hand-written now. A regression there is silent -- a homography-guided
    matcher simply keeps no matches -- so it is checked rather than assumed.
    """
    state = np.random.RandomState(7)
    source = state.uniform(20.0, 460.0, size=(40, 2))
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])

    target = np.array([(matrix @ np.array([p[0], p[1], 1.0]))[:2]
                       for p in source])
    target[::4] += 60.0

    points1 = [np.array(p) for p in source]
    points2 = [np.array(p) for p in target]

    estimated, inliers = create(
        "EstimateHomography", "ocv").estimate(points1, points2, 1.0)

    assert estimated is not None
    assert len(inliers) == len(points1)
    assert sum(inliers) == 30, "the ten displaced pairs should be outliers"
    assert not any(inliers[::4])


def test_too_few_points_is_refused_rather_than_asserted():
    """The C++ checked the count and logged; OpenCV would have asserted."""
    points = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]

    estimated, inliers = create(
        "EstimateHomography", "ocv").estimate(points, points, 1.0)
    assert estimated is None
    assert inliers == []

    estimated, inliers = create(
        "EstimateFundamentalMatrix", "ocv").estimate(points, points, 1.0)
    assert estimated is None
    assert inliers == []


def test_the_matcher_refuses_an_empty_side():
    from kwiver.vital.types import DescriptorSet, SimpleFeatureSet

    matcher = create("MatchFeatures", "ocv_flann_based")
    empty = DescriptorSet([])
    features = SimpleFeatureSet([])

    assert matcher.match(features, empty, features, empty) is None


def test_the_matcher_rejects_a_zero_cross_check_k():
    from kwiver.vital.config import empty_config

    matcher = create("MatchFeatures", "ocv_flann_based")
    cfg = matcher.get_configuration()
    cfg.set_value("cross_check_k", "0")

    assert not matcher.check_configuration(cfg)


def test_the_fundamental_estimator_rejects_a_confidence_out_of_range():
    algorithm = create("EstimateFundamentalMatrix", "ocv")

    for value, valid in (("0.99", True), ("1.0", True), ("0", False),
                         ("1.5", False)):
        cfg = algorithm.get_configuration()
        cfg.set_value("confidence_threshold", value)
        assert algorithm.check_configuration(cfg) is valid, value
