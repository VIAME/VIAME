"""What the feature chain records, for the OpenCV implementations P7-T04 replaces.

P7-T01 recorded the filters and the detectors and missed this corner: SIFT,
SURF, the FLANN matcher and the two RANSAC estimators are OpenCV-backed
`library/image_processing` names with no recording at all. They cannot be
replaced without one, so this is recorded first, exactly as everything else
in phase 7 was.

Four kinds, in the order the chain runs them:

* `features` -- `detect_features` and then `extract_descriptors` of the same
  name, on the same image, which is how `track_features:core` uses them. The
  pair is recorded together on purpose: the extractor takes the feature set
  by reference and may replace it, and a SIFT descriptor depends on the
  octave the detector found the keypoint in, which travels inside the
  implementation's own feature set and not through `vital::feature`. Only a
  recording of the two together says whether that survived a port.
* `matches` -- `match_features` over two images' descriptors, recorded as
  index pairs.
* `homography` and `fundamental` -- the two estimators over the matched
  points, recorded as the 3x3 matrix.

The `inliers` out-parameter of both estimators cannot be recorded: it is a
`std::vector<bool>&` and the generated pybind11 binding hands python a copy,
so nothing a python caller does can observe it. That is a hole in the
recording, and it is why the trampolines are hand-written before the port --
see `design/lite-findings.md`.
"""

# Real frames rather than the synthetic fixtures: a gradient or a bar chart
# has no corners, and a detector recorded on one says nothing. `rgb8` is here
# for the small-image path, where SURF finds nothing and the recording pins
# that too.
IMAGES = ("frame_00", "frame_01", "rgb8")

# The image pair the matcher and the estimators run over. Consecutive frames
# of the same clip, so the correspondence is real and a homography is close
# to the identity.
PAIR = ("frame_00", "frame_01")

# `detect_features` and `extract_descriptors` share these names and their
# config keys, which is why one table drives both.
FEATURES = {
    "ocv_SIFT": [
        ("defaults", {}),
        # The two keys that change how many features come out and how big
        # each one's support is. `n_features` also exercises the retain-best
        # path, which sorts by response and truncates.
        ("few", {"n_features": "20"}),
        ("coarse", {"n_octave_layers": "2", "contrast_threshold": "0.08"}),
    ],
    "ocv_SURF": [
        ("defaults", {}),
        # Extended descriptors are 128 wide rather than 64, and upright skips
        # the orientation pass -- the two options that change the descriptor
        # rather than merely how many there are.
        ("extended", {"extended": "true"}),
        ("upright_500", {"upright": "true", "hessian_threshold": "500"}),
    ],
}

# The matcher, over each detector's descriptors. `cross_check` off is the
# other half of the implementation: it takes the single nearest neighbour
# rather than requiring agreement in both directions.
MATCHERS = {
    "ocv_flann_based": [
        ("defaults", {}),
        ("no_cross_check", {"cross_check": "false"}),
    ],
    # C++, and staying C++ -- but it nests `estimate_homography` and keeps
    # only the matches that estimator calls inliers, so it is the one case
    # that can see the `inliers` out parameter at all. Recorded here, with
    # the OpenCV estimator underneath, it is what says the out parameter
    # still arrives once that estimator is python: if it does not, every
    # match is dropped and the recorded count goes to zero.
    "homography_guided": [
        ("ocv_estimator", {
            "feature_matcher1:type": "ocv_flann_based",
            "homography_estimator:type": "ocv",
            "inlier_scale": "10",
        }),
    ],
    # `match_features:fundamental_matrix_guided` would be the same test for
    # `estimate_fundamental_matrix`, but P5-T04 removed it -- nothing
    # selects it. That estimator has no C++ consumer left in the tree, so
    # its out parameter is checked by a unit test rather than by a golden.
}

# Which detector's descriptors the matcher and the estimators run over.
MATCH_FEATURES = ("ocv_SIFT", "ocv_SURF")

# `track_features:core` over the pair, which is what the shipped stabilizer
# runs. It is here because it is the only C++ caller of
# `extract_descriptors` in the tree: it hands the detector's feature set to
# the extractor and then indexes the descriptors by the feature set that
# comes back. If a python extractor cannot replace that set, every descriptor
# is paired with the wrong feature and the tracks are nonsense -- which is
# invisible to a `features` case, where python is the caller and gets the
# replacement back as a return value.
TRACKERS = {
    "core": [
        ("ocv_SIFT", {
            "feature_detector:type": "ocv_SIFT",
            "descriptor_extractor:type": "ocv_SIFT",
            "feature_matcher:type": "ocv_flann_based",
        }),
        ("ocv_SURF", {
            "feature_detector:type": "ocv_SURF",
            "descriptor_extractor:type": "ocv_SURF",
            "feature_matcher:type": "ocv_flann_based",
        }),
    ],
}

# ----------------------------------------------------------------------------
# The two estimators
# ----------------------------------------------------------------------------
#
# On synthetic correspondences rather than on matched features. Both are
# RANSAC, and RANSAC in OpenCV is deterministic for a given input -- its
# sampler has a fixed seed -- so a fixed correspondence set makes these exact
# recordings. Feeding them `ocv_flann_based` matches instead does not: the
# FLANN matcher is randomised (see UNSTABLE below), one match in or out
# changes which points RANSAC sees, and the estimated matrix moved by 29% for
# the homography and 52% for the fundamental matrix between runs of the same
# code. A recording that loose says nothing about a port.
#
# It also puts the estimator on its own, which is what wants testing here:
# the matcher already has cases of its own.
SYNTHETIC_SEED = 20260910

# A planar scene: points on a plane, mapped by a known homography, with every
# fifth correspondence displaced to make an outlier RANSAC has to reject.
HOMOGRAPHY_POINTS = 60
HOMOGRAPHY_OUTLIER_STRIDE = 5
HOMOGRAPHY_OUTLIER_SPREAD = 40.0
HOMOGRAPHY_TRUE = (
    (1.02, -0.03, 7.0),
    (0.015, 0.99, -4.0),
    (2e-5, -1e-5, 1.0),
)

# A non-planar scene, because a fundamental matrix is degenerate on a plane:
# 3D points in a box, seen by two cameras that differ by a small rotation and
# a mostly sideways translation.
FUNDAMENTAL_POINTS = 80
FUNDAMENTAL_DEPTH = 5.0
FUNDAMENTAL_FOCAL = 500.0
FUNDAMENTAL_CENTRE = (240.0, 135.0)
FUNDAMENTAL_ROTATION = (0.02, 0.03, 0.01)
FUNDAMENTAL_TRANSLATION = (0.5, 0.05, 0.1)
FUNDAMENTAL_OUTLIER_STRIDE = 8
FUNDAMENTAL_OUTLIER_SPREAD = 30.0

ESTIMATE_HOMOGRAPHY = {
    "ocv": [
        ("defaults", {}),
    ],
}

ESTIMATE_FUNDAMENTAL = {
    "ocv": [
        ("defaults", {}),
        ("confidence_90", {"confidence_threshold": "0.90"}),
    ],
}

# The estimators take an error tolerance rather than a config key for it. One
# tight enough to reject the outliers and one loose enough to admit them.
INLIER_SCALES = (1.0, 10.0)


# ----------------------------------------------------------------------------
# What cannot be recorded exactly
# ----------------------------------------------------------------------------
#
# `cv::FlannBasedMatcher` builds randomised KD-trees, and OpenCV seeds them
# from the clock, so `ocv_flann_based` gives a different answer on every call
# -- 45 or 46 matches out of the same 81 descriptors, within one process.
# Everything downstream of it inherits that.
#
# So these cases are contracts on agreement rather than on bytes. The
# thresholds are what eight runs of the recorded code stayed inside, with
# room: the matcher agreed with its own recording on 97.8% of pairs at worst
# and never moved the count by more than one.
UNSTABLE = {
    ("matches", "ocv_flann_based"):
        "cv::FlannBasedMatcher builds randomised KD-trees seeded from the "
        "clock, so the match set differs run to run",
    ("matches", "homography_guided"):
        "its nested feature_matcher1 is ocv_flann_based, which is randomised",
    ("tracks", "core"):
        "its feature_matcher is ocv_flann_based, which is randomised; the "
        "feature locations are exact and only the linking varies",
}

# For an unstable `matches` case: how far the count may move, and how much of
# the recorded set must still be there.
MATCH_COUNT_TOLERANCE = 0.10
MATCH_AGREEMENT = 0.90

# For an unstable `tracks` case: the (frame, x, y) locations are the
# detector's and must be exact -- they are also what a mis-paired descriptor
# would leave untouched -- while how many of them link into a two-frame track
# is the matcher's and may move.
TRACK_LENGTH_TOLERANCE = 0.10


def unstable(kind, impl, variant=None):
    """Why this case cannot be compared exactly, or None."""
    return UNSTABLE.get((kind, impl))


# ----------------------------------------------------------------------------
# Where the port deliberately does not reproduce the recording
# ----------------------------------------------------------------------------
#
# **The C++ SIFT and SURF wrappers ignored their configuration.** All five of
# each one's config keys, silently: every non-default variant recorded here
# came out byte-identical to `defaults` -- 81 SIFT features whether
# `n_features` says 0 or 20, 64-wide SURF descriptors with `extended` set.
#
# The cause is one line, repeated in all four wrappers:
#
#     detector.constCast< cv::FeatureDetector >() = create( ... );
#
# `cv::Ptr::constCast` returns a new `Ptr` by value, so the assignment
# replaces a temporary and the freshly built detector is destroyed on the
# next line. The wrapper reran it before every call, and every call used the
# detector built at construction from the defaults.
#
# The port honours the configuration. Reproducing the defect would mean
# implementing five documented keys per algorithm that do nothing, and
# nothing can be depending on them doing nothing: `common_image_stabilizer`
# and `utility_register_frames_3-cam` both ask for `hessian_threshold =
# 5000` and `upright = true` and have been getting 100 and false. The `few`
# and `coarse` variants are what the recorded code should have produced and
# the port does.
#
# `defaults` is held to the recording exactly, which is what says the port is
# the same algorithm.
DIVERGENCES = {
    ("ocv_SIFT", "few"):
        "the recorded wrapper ignored n_features; the port applies it",
    ("ocv_SIFT", "coarse"):
        "the recorded wrapper ignored n_octave_layers and "
        "contrast_threshold; the port applies them",
    ("ocv_SURF", "extended"):
        "the recorded wrapper ignored extended, so its descriptors stayed 64 "
        "wide; the port applies it and they are 128",
    ("ocv_SURF", "upright_500"):
        "the recorded wrapper ignored upright and hessian_threshold; the "
        "port applies them",
}


def divergence_reason(impl, variant, input_name=None):
    """Why the port is not held to the recording for this case, or None."""
    return DIVERGENCES.get((impl, variant))


def unstable_reason(impl, variant, name=None):
    """The per-input hook the other case files have; nothing uses it here."""
    return None
