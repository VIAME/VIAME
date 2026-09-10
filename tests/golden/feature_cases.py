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
}

# Which detector's descriptors the matcher and the estimators run over.
MATCH_FEATURES = ("ocv_SIFT", "ocv_SURF")

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

# The estimators take an error tolerance rather than a config key for it.
INLIER_SCALES = (1.0, 10.0)


def unstable_reason(impl, variant, name=None):
    """Nothing here is unstable; the signature matches the other case files."""
    return None
