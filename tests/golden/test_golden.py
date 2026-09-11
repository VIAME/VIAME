"""Replay every recorded golden against the implementations this build has.

A recording is the behaviour of the implementation that is being replaced.
Once a replacement registers the old name, this test is what says the
replacement behaves the same: same shape, same dtype, and values within the
tolerance stated for that implementation below.

Run just these:  ctest -L GOLDEN
Re-record:       see tests/golden/README.md
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import cases as case_spec           # noqa: E402
import calib_cases                  # noqa: E402
import calib_runner                 # noqa: E402
import codec_cases                  # noqa: E402
import feature_cases                # noqa: E402
import feature_runner               # noqa: E402
import imageio_utils                # noqa: E402
import measurement_cases            # noqa: E402
import measurement_runner           # noqa: E402
import measurement_fixtures         # noqa: E402
import opencv_cases                 # noqa: E402
import pipeline_runner              # noqa: E402
import refine_cases                 # noqa: E402
import refine_runner                # noqa: E402
import warp_cases                   # noqa: E402
import warp_runner                  # noqa: E402
import runner                       # noqa: E402


# Per implementation: (max absolute difference, max mean absolute difference).
#
# Everything is exact while the recording and the build are the same code.
# A replacement that cannot be bit exact raises its own entry here, together
# with the reason, rather than loosening the default for everyone.
TOLERANCES = {
    "__default__": (0.0, 0.0),
    # `ocv_convert_color` since P7-T04b. OpenCV's colour conversions work in
    # fixed point for an 8 bit image -- a hue through a reciprocal table, an
    # L*a*b* through a cube-root table -- where `image_ops` works in double,
    # so the two round apart on the last count. Measured over the eight
    # recorded pairs: never more than 1 count, and 0.33 mean at worst
    # (hsv_to_rgb, where the inverse of a quantised hue lands between two
    # bytes on a third of the pixels). Two of the eight are exact.
    #
    # Reproducing the tables instead was considered in P7-T03 and not done:
    # they are a precision compromise for speed, and a port that is more
    # accurate than what it replaces is the better of the two to keep.
    "ocv_convert_color": (1.0, 0.5),
}

# The same, for a name that means different things in different kinds. `ocv`
# is a split_image, a merge_images, a draw and an image_io as well as a warp,
# and only the warp needs any room at all.
TOLERANCES_BY_KIND = {
    # `cv::warpPerspective` interpolates in fixed point at INTER_BITS 5,
    # quantising the sample position to a thirty-second of a pixel, where
    # `image_ops` interpolates in double. P7-T03 measured both against the
    # exact bilinear answer and found `image_ops` the more accurate of the
    # two -- 0.254 mean error against OpenCV's 0.366 -- so the tolerance is
    # here rather than in a reproduction of the quantisation. Measured over
    # the eight recorded warps: 4 at worst, 0.21 mean, and the four cases
    # whose homography is an integer translation are exact.
    ( "warp", "ocv" ): (4.0, 0.25),
}



REMOVED_PATH = os.path.join(HERE, "..", "baseline", "removed.json")


# The algorithm interface each kind of case exercises. A pipeline is not an
# algorithm and has no interface, which is why it is absent rather than None.
INTERFACE_OF_KIND = {
    "image_filter": "image_filter",
    "image_io": "image_io",
    "decode": "image_io",
    "round_trip": "image_io",
    "split_image": "split_image",
    "detect_motion": "detect_motion",
    "detect": "image_object_detector",
    # The feature chain: `features` runs a detect_features and the
    # extract_descriptors of the same name, so it is keyed on the detector
    # -- a removal would take both halves together.
    "disparity": "compute_stereo_depth_map",
    "features": "detect_features",
    "refine": "refine_detections",
    "warp": "warp_image",
    "matches": "match_features",
    "tracks": "track_features",
    "homography": "estimate_homography",
    "fundamental": "estimate_fundamental_matrix",
}


def removed_names():
    """Implementations deliberately removed, from the baseline's removed.json.

    A recording of a name that is gone on purpose is history, not a contract.
    Skipping here rather than failing keeps the recording available for
    comparison if the decision is ever revisited.

    Keyed by (name, interface), not by name. A name means nothing on its own:
    `ocv` is registered for eleven interfaces and `vxl` for nine, and phase 5
    removed some of each while others stayed. Keying by name alone silently
    skipped the twelve `vxl` image_io cases -- the reader that is still
    registered and still has to reproduce its recording -- because a `vxl`
    bundle_adjust nobody used had been removed.
    """
    if not os.path.exists(REMOVED_PATH):
        return {}

    with open(REMOVED_PATH) as handle:
        return {(entry["name"], entry["interface"]): entry.get("reason", "")
                for entry in json.load(handle)}


def load_manifest(group):
    path = os.path.join(HERE, group, "manifest.json")

    if not os.path.exists(path):
        return None

    with open(path) as handle:
        manifest = json.load(handle)

    # Other recordings live here too, such as the video reader data phase 4
    # consumes; only a manifest with cases is one of these goldens
    return manifest if isinstance(manifest.get("cases"), list) else None


# The replacement table each group uses. Keyed by group because a name means
# different things in different ones: `ocv` is the image_io the codecs group
# records, and the same string is a split_image and a merge_images elsewhere.
REPLACEMENTS_BY_GROUP = {
    "codecs": codec_cases.REPLACEMENTS,
}


# Where a documented divergence is declared, per kind. `cases.py` holds the
# ones phase 3 found; the feature chain's are in `feature_cases.py`, beside
# the cases themselves.
# Where an unstable case's reason and comparison live, per kind. A module
# here provides `unstable(kind, impl, variant)` and `compare_unstable`.
UNSTABLE_OF_KIND = {
    "matches": (feature_cases, feature_runner),
    "tracks": (feature_cases, feature_runner),
    "calibration_pipeline": (measurement_cases, measurement_cases),
    "measurement": (measurement_cases, measurement_cases),
}


DIVERGENCES_OF_KIND = {
    "disparity": measurement_cases,
    "refine": refine_cases,
    "warp": warp_cases,
    "features": feature_cases,
    "matches": feature_cases,
    "tracks": feature_cases,
    "homography": feature_cases,
    "fundamental": feature_cases,
}


def divergence_reason(case, input_name=None):
    source = DIVERGENCES_OF_KIND.get(case["kind"], case_spec)
    return source.divergence_reason(case["impl"], case["variant"], input_name)


def collect_cases():
    cases = []

    for group in sorted(os.listdir(HERE)):
        manifest = load_manifest(group)

        if manifest is None:
            continue

        replacements = REPLACEMENTS_BY_GROUP.get(group, case_spec.REPLACEMENTS)

        for case in manifest["cases"]:
            cases.append((group, case, case["impl"]))

            replacement = replacements.get(case["impl"])
            if replacement:
                cases.append((group, case, replacement))

    return cases


def case_id(item):
    group, case, impl = item
    suffix = "" if impl == case["impl"] else "->" + impl
    return "{}:{}:{}{}:{}".format(group, case["kind"], case["impl"], suffix,
                                  case["variant"])


@pytest.fixture(scope="session", autouse=True)
def modules():
    runner.load_modules()


def input_path(name):
    for ext in (".png", ".npz", ".npy"):
        candidate = os.path.join(HERE, "inputs", name + ext)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("no fixture named '{}'".format(name))


def container_path(name):
    """A codec fixture: an encoded file rather than an array."""
    import codec_fixtures

    directory = os.path.join(HERE, "inputs", codec_cases.INPUT_SUBDIR)

    for candidate, path in codec_fixtures.paths(directory):
        if candidate == name:
            return path

    raise FileNotFoundError("no codec fixture named '{}'".format(name))


def _first_index(recorded_names):
    """Order the split inputs the way the recording lists their pieces."""
    order = {}
    for position, name in enumerate(recorded_names):
        order.setdefault(name.rsplit("_", 1)[0], position)
    return lambda name: order[name]


def run_case(case, impl):
    if case["kind"] == "image_filter":
        arrays = [imageio_utils.load(input_path(name)) for name in case["inputs"]]
        return runner.run_image_filter(impl, case["config"], arrays)

    if case["kind"] == "image_io":
        paths = [input_path(name) for name in case["inputs"]]
        return runner.run_image_io_load(impl, case["config"], paths)

    if case["kind"] == "decode":
        paths = [container_path(name) for name in case["inputs"]]
        return runner.run_image_io_load(impl, case["config"], paths)

    if case["kind"] == "round_trip":
        arrays = [imageio_utils.load(input_path(name)) for name in case["inputs"]]
        with tempfile.TemporaryDirectory() as work_dir:
            return runner.run_image_io_save_load(
                impl, case["config"], arrays, case["extension"], work_dir)

    if case["kind"] == "split_image":
        arrays = [imageio_utils.load(input_path(name))
                  for name in sorted({name.rsplit("_", 1)[0]
                                      for name in case["inputs"]},
                                     key=_first_index(case["inputs"]))]
        split = runner.run_split_image(impl, case["config"], arrays)
        return [piece for pieces in split for piece in pieces]

    if case["kind"] == "detect_motion":
        arrays = [imageio_utils.load(input_path(name)) for name in case["inputs"]]
        return runner.run_detect_motion(impl, case["config"], arrays)

    if case["kind"] == "detect":
        arrays = [imageio_utils.load(input_path(name)) for name in case["inputs"]]
        return runner.run_image_object_detector(impl, case["config"], arrays)

    if case["kind"] == "disparity":
        left, right = (imageio_utils.load(input_path(name))
                       for name in measurement_cases.STEREO)
        return [runner.run_stereo_depth_map(impl, case["config"], left, right)]

    if case["kind"] == "calibration_pipeline":
        left, right = measurement_cases.calibration_view_names()
        outputs = measurement_runner.run_stereo_pipeline(
            impl, left, right, tuple(case["settings"]))
        return [measurement_runner.calibration_arrays(outputs)]

    if case["kind"] == "measurement":
        return [measurement_runner.run_measurement_pipeline(
            impl, tuple(case["settings"]), case.get("paired", False))]

    if case["kind"] == "mono_calibration":
        names = measurement_cases.mono_calibration_view_names()
        return [measurement_runner.run_mono_pipeline(
            impl, names, tuple(case["settings"]))]

    if case["kind"] == "warp":
        source = imageio_utils.load(input_path(warp_cases.SOURCE))
        destination = imageio_utils.load(input_path(warp_cases.DESTINATION))
        mask = imageio_utils.load(input_path(warp_cases.MASK))
        return [warp_runner.run(impl, case["variant"], source, destination,
                                mask)]

    if case["kind"] == "refine":
        array = imageio_utils.load(input_path(refine_cases.IMAGE))
        return [refine_runner.run(impl, case["config"], array)]

    if case["kind"] == "features":
        arrays = [imageio_utils.load(input_path(name))
                  for name in case["inputs"]]
        return [feature_runner.detect_and_extract(impl, case["config"], array)
                for array in arrays]

    if case["kind"] == "matches":
        arrays = [imageio_utils.load(input_path(name))
                  for name in feature_cases.PAIR]
        return [feature_runner.match(impl, case["config"], case["features"],
                                     {}, arrays)]

    if case["kind"] == "tracks":
        arrays = [imageio_utils.load(input_path(name))
                  for name in feature_cases.PAIR]
        return [feature_runner.track(impl, case["config"], arrays)]

    if case["kind"] in ("homography", "fundamental"):
        estimate = (feature_runner.estimate_homography
                    if case["kind"] == "homography"
                    else feature_runner.estimate_fundamental)
        return [estimate(impl, case["config"], case["inlier_scale"])]

    if case["kind"] == "calibration":
        return [calib_runner.load_calibration(
                    os.path.join(REPO_ROOT, calib_cases.CALIBRATIONS[name]))
                for name in case["inputs"]]

    if case["kind"] == "nodes":
        return [calib_runner.dump_document(
                    os.path.join(REPO_ROOT, calib_cases.DOCUMENTS[name]))
                for name in case["inputs"]]

    if case["kind"] == "process_pipeline":
        return pipeline_runner.run_detections(
            case["pipeline"], case.get("settings", ()))

    if case["kind"] == "pipeline":
        outputs = pipeline_runner.run(impl)
        missing = sorted(set(case["outputs"]) - set(outputs))
        assert not missing, "{} no longer wrote: {}".format(
            case["impl"], ", ".join(missing))
        return [outputs[name] for name in case["outputs"]]

    raise AssertionError("unknown case kind '{}'".format(case["kind"]))


def check_refusals(item, case, impl):
    """An input the recording says the implementation rejects must still be.

    A replacement that quietly started accepting a single channel image where
    the recorded one threw would be a change in behaviour that no output
    comparison could see, because there is no output to compare.
    """
    refuses = case.get("refuses") or {}

    if not refuses:
        return

    if not opencv_cases.refusal_is_reliable(case["impl"], case["variant"]):
        return

    for name, reason in sorted(refuses.items()):
        probe = dict(case, inputs=[name], outputs={})
        with pytest.raises(Exception):
            run_case(probe, impl)


# How far a recorded detection score may move. The geometry is integer
# valued and has to be exact; a score is a float sum, and OpenCV accumulates
# a box filter in float where `image_ops` accumulates in double, so the two
# agree to about seven digits rather than to the bit. Relative, because the
# scores here range from a fraction to seventy.
SCORE_TOLERANCE = 1e-6


def check_detections(item, case, outputs, group):
    """A recording whose values are detections rather than pixels."""
    for name, actual in zip(case["inputs"], outputs):
        record = case["outputs"][name]

        with open(os.path.join(HERE, group, record["file"])) as handle:
            expected = json.load(handle)

        assert len(actual) == len(expected), (
            "{} {}: {} detections, recorded {}".format(
                case_id(item), name, len(actual), len(expected)))

        for index, (got, want) in enumerate(zip(actual, expected)):
            assert sorted(got) == sorted(want), (
                "{} {} detection {}: fields {} != recorded {}".format(
                    case_id(item), name, index, sorted(got), sorted(want)))

            # The box: exact, because it is a pixel rectangle
            assert got["bbox"] == want["bbox"], (
                "{} {} detection {}: bbox is {!r}, recorded {!r}".format(
                    case_id(item), name, index, got["bbox"], want["bbox"]))

            def close(a, b):
                return abs(a - b) <= SCORE_TOLERANCE * max(1.0, abs(b))

            assert close(got["confidence"], want["confidence"]), (
                "{} {} detection {}: confidence is {!r}, recorded "
                "{!r}".format(case_id(item), name, index, got["confidence"],
                              want["confidence"]))

            for label, score in sorted(want.get("types", {}).items()):
                assert label in got.get("types", {}), (
                    "{} {} detection {}: no score for '{}'".format(
                        case_id(item), name, index, label))
                assert close(got["types"][label], score), (
                    "{} {} detection {} type '{}': {!r}, recorded "
                    "{!r}".format(case_id(item), name, index, label,
                                  got["types"][label], score))


# How far a recorded feature or descriptor may move. Zero: SIFT and SURF are
# the same OpenCV code before and after the port, so a difference here is a
# difference in what the wrapper hands them or hands back, which is the only
# thing the port can get wrong. A tolerance would hide exactly that.
ARRAY_TOLERANCE = 0.0


def check_array_case(item, case, outputs, group):
    """A recording whose values are named arrays: features, descriptors, matches.

    Each member is compared for shape, dtype and value. Shape first, because
    a detector that finds a different number of features is a different
    detector and saying "12 rows, recorded 81" is more use than a value
    mismatch on row zero.
    """
    for name, actual in zip(case["inputs"], outputs):
        record = case["outputs"][name]
        expected = feature_runner.load(os.path.join(HERE, group, record["file"]))

        assert sorted(actual) == sorted(expected), (
            "{} {}: members {} != recorded {}".format(
                case_id(item), name, sorted(actual), sorted(expected)))

        cases_module, runner_module = UNSTABLE_OF_KIND.get(
            case["kind"], (None, None))
        unstable = (cases_module.unstable(case["kind"], case["impl"],
                                          case["variant"])
                    if cases_module else None)

        for member in sorted(expected):
            got = np.asarray(actual[member])
            want = np.asarray(expected[member])

            if unstable:
                problems = runner_module.compare_unstable(
                    case["kind"], member, got, want)
                assert not problems, "{} {} '{}' ({}): {}".format(
                    case_id(item), name, member, unstable,
                    "; ".join(problems))
                continue

            assert got.shape == want.shape, (
                "{} {} '{}': shape {} != recorded {}".format(
                    case_id(item), name, member, got.shape, want.shape))

            if want.size == 0:
                continue

            difference = np.abs(got.astype(np.float64) -
                                want.astype(np.float64))

            # Zero unless the case's own module says otherwise. The
            # `measurement` kind's two rectified variants do, and
            # `measurement_cases.py` says why.
            relative = (measurement_cases.array_tolerance(
                            case["impl"], case["variant"])
                        if case["kind"] == "measurement" else 0.0)

            allowed = np.maximum(
                ARRAY_TOLERANCE,
                relative * np.maximum(1.0, np.abs(want.astype(np.float64))))

            assert bool(np.all(difference <= allowed)), (
                "{} {} '{}': max difference {} exceeds {}".format(
                    case_id(item), name, member, difference.max(),
                    float(np.max(allowed))))


def check_calibration_truth(item, arrays):
    """The calibration against the rig the views were rendered through.

    The one case in this framework with a right answer rather than only a
    previous answer: `measurement_fixtures.py` renders the views through a
    known stereo rig, so this can say not merely "different from the
    recording" but "and the recording was correct". A port that reproduces
    the recording exactly and a port that is right are the same thing here,
    and if they ever stop being, this is what says which broke.
    """
    truth = measurement_cases.CALIBRATION_TRUTH
    tolerances = measurement_cases.CALIBRATION_TOLERANCES

    def close(name, actual, expected, tolerance):
        assert abs(actual - expected) <= tolerance * abs(expected), (
            "{} {}: {} against a true {}, more than {:.1%} out".format(
                case_id(item), name, actual, expected, tolerance))

    for side, key in (("left", "M1"), ("right", "M2")):
        matrix = arrays[key]

        close("fx_" + side, matrix[0][0], truth["fx_" + side],
              tolerances["focal"])
        close("fy_" + side, matrix[1][1], truth["fy_" + side],
              tolerances["focal"])
        close("cx_" + side, matrix[0][2], truth["cx_" + side],
              tolerances["centre"])
        close("cy_" + side, matrix[1][2], truth["cy_" + side],
              tolerances["centre"])

    baseline = float(np.linalg.norm(arrays["T"]))
    close("baseline", baseline, truth["baseline"], tolerances["baseline"])

    # The rig has no lens distortion, so a calibration that finds some is
    # fitting noise or has its model wrong.
    for key in ("D1", "D2"):
        worst = float(np.abs(arrays[key]).max())
        assert worst <= measurement_cases.CALIBRATION_MAX_DISTORTION, (
            "{} {}: distortion up to {} on a distortion free rig".format(
                case_id(item), key, worst))


def check_mono_calibration_truth(item, arrays):
    """The single camera calibration against the rig its views came from.

    The left half of `check_calibration_truth`, on the same synthetic views
    and at the same tolerances. There is no baseline and no second camera, so
    what is left is the intrinsics and the "no distortion on a distortion
    free rig" check.
    """
    truth = measurement_cases.MONO_CALIBRATION_TRUTH
    tolerances = measurement_cases.CALIBRATION_TOLERANCES

    def close(name, actual, expected, tolerance):
        assert abs(actual - expected) <= tolerance * abs(expected), (
            "{} {}: {} against a true {}, more than {:.1%} out".format(
                case_id(item), name, actual, expected, tolerance))

    for key in ("fx", "fy"):
        close(key, float(arrays[key][0][0]), truth[key], tolerances["focal"])

    for key in ("cx", "cy"):
        close(key, float(arrays[key][0][0]), truth[key], tolerances["centre"])

    for key in ("k1", "k2", "k3", "p1", "p2"):
        worst = float(np.abs(arrays[key]).max())
        assert worst <= measurement_cases.CALIBRATION_MAX_DISTORTION, (
            "{} {}: distortion of {} on a distortion free rig".format(
                case_id(item), key, worst))


def check_measurement_truth(item, case, arrays):
    """The measured lengths against the segments the scene actually holds.

    The scene draws five segments of known length on a plane at a known
    depth, so this says "and the recording was right" as well as "and the
    port reproduced it". Finding 1.20 is what it is for: every length on
    this branch was zero, and no recording existed to say so.

    `measurement_cases.MEASUREMENT_LENGTH_TOLERANCE` gives `None` for a
    method that is not supposed to be accurate -- `depth_projection` is told
    the wrong depth by construction -- and those are checked only for being
    a positive number of the right order, which still catches zero.
    """
    truth = measurement_fixtures.measurement_truth()
    tolerance = measurement_cases.MEASUREMENT_LENGTH_TOLERANCE.get(
        case["variant"], None)

    ids = [int(round(value)) for value in arrays["track_ids"].reshape(-1)]
    lengths = arrays["length"].reshape(-1)

    for identifier, length in zip(ids, lengths):
        expected = truth[identifier]["length_mm"]

        assert length > 0.0, (
            "{} track {}: measured {}, and the segment is {:.1f} mm "
            "long".format(case_id(item), identifier, length, expected))

        if tolerance is None:
            continue

        assert abs(length - expected) <= tolerance * expected, (
            "{} track {}: measured {:.3f} mm against a true {:.3f}, more "
            "than {:.1%} out".format(
                case_id(item), identifier, length, expected, tolerance))


def check_json_case(item, case, outputs, group):
    """A recording whose values are parsed structure rather than pixels.

    Compared exactly: these are text files holding decimal literals, so a
    parser that rounds differently is a parser that is wrong. Compared
    structurally rather than by digest, so a failure says which node moved.
    """
    for name, actual in zip(case["inputs"], outputs):
        record = case["outputs"][name]

        with open(os.path.join(HERE, group, record["file"])) as handle:
            expected = json.load(handle)

        if case["kind"] == "calibration":
            actual = {key: value.tolist() for key, value in actual.items()}
            actual = {key: actual[key] for key in calib_cases.CALIBRATION_KEYS}

        assert sorted(actual) == sorted(expected), (
            "{} {}: nodes {} != recorded {}".format(
                case_id(item), name, sorted(actual), sorted(expected)))

        for key in sorted(expected):
            assert actual[key] == expected[key], (
                "{} {}: node '{}' is {!r}, recorded {!r}".format(
                    case_id(item), name, key, actual[key], expected[key]))


REMOVED = removed_names()


@pytest.mark.parametrize("item", collect_cases(), ids=case_id)
def test_golden(item):
    group, case, impl = item

    interface = INTERFACE_OF_KIND.get(case["kind"])
    removal = REMOVED.get((case["impl"], interface)) if interface else None

    if removal:
        pytest.skip("{} was removed as an {} on purpose: {}".format(
            case["impl"], interface, removal))

    if impl != case["impl"] and not runner.is_registered( case["kind"], impl ):
        pytest.skip("{} is not registered in this build".format(impl))

    # A divergence is declared by hand and only ever for a case the port
    # deliberately does not reproduce, so it applies whether the port runs
    # under the recorded name or under a replacement's. `replacing` used to
    # gate this, which meant a name reimplemented in place rather than
    # aliased -- ocv_SIFT, since P7-T04 -- could not declare one.
    whole_case = divergence_reason(case)

    if whole_case:
        # Still run it, so a crash or a refused config is caught
        run_case(case, impl)
        pytest.skip("deliberate divergence: {}".format(whole_case))

    check_refusals(item, case, impl)

    outputs = run_case(case, impl)

    if case["kind"] in ("detect", "process_pipeline"):
        check_detections(item, case, outputs, group)
        return

    if case["kind"] == "calibration_pipeline":
        check_calibration_truth(item, outputs[0])
        check_array_case(item, case, outputs, group)
        return

    if case["kind"] == "mono_calibration":
        check_mono_calibration_truth(item, outputs[0])
        check_array_case(item, case, outputs, group)
        return

    if case["kind"] == "measurement":
        check_measurement_truth(item, case, outputs[0])
        check_array_case(item, case, outputs, group)
        return

    if case["kind"] in ("features", "matches", "tracks", "homography",
                        "fundamental", "refine"):
        check_array_case(item, case, outputs, group)
        return

    if case["kind"] in ("calibration", "nodes"):
        check_json_case(item, case, outputs, group)
        return

    max_tol, mean_tol = TOLERANCES_BY_KIND.get(
        ( case["kind"], impl ),
        TOLERANCES.get(impl, TOLERANCES["__default__"]))
    unstable = case.get("unstable", {})

    names = (list(case["outputs"]) if case["kind"] == "pipeline"
             else case["inputs"])

    for name, actual in zip(names, outputs):
        # A path the recording left uninitialised, which the replacement
        # deliberately implements properly instead. cases.py says why
        if divergence_reason(case, name):
            continue

        # A lossy container is compared decoder to decoder, at the tolerance
        # codec_cases states for it, rather than at this implementation's.
        if case["kind"] == "decode":
            max_tol, mean_tol = codec_cases.tolerance(name)

        record = case["outputs"][name]
        expected = imageio_utils.load(os.path.join(HERE, group, record["file"]))

        # A single channel result is stored two dimensional
        if actual.ndim == 3 and actual.shape[2] == 1:
            actual = actual[:, :, 0]

        assert list(actual.shape) == list(expected.shape), (
            "{} {}: shape {} != recorded {}".format(
                case_id(item), name, actual.shape, expected.shape))
        assert str(actual.dtype) == str(expected.dtype), (
            "{} {}: dtype {} != recorded {}".format(
                case_id(item), name, actual.dtype, expected.dtype))

        # The recording itself could not be reproduced when it was taken, so
        # only its shape and dtype are a contract. cases.py says why
        if name in unstable:
            continue

        difference = np.abs(actual.astype(np.float64)
                            - expected.astype(np.float64))

        assert difference.max() <= max_tol, (
            "{} {}: max abs diff {} exceeds {}".format(
                case_id(item), name, difference.max(), max_tol))
        assert difference.mean() <= mean_tol, (
            "{} {}: mean abs diff {} exceeds {}".format(
                case_id(item), name, difference.mean(), mean_tol))
