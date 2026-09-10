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
import opencv_cases                 # noqa: E402
import pipeline_runner              # noqa: E402
import runner                       # noqa: E402


# Per implementation: (max absolute difference, max mean absolute difference).
#
# Everything is exact while the recording and the build are the same code.
# A replacement that cannot be bit exact raises its own entry here, together
# with the reason, rather than loosening the default for everyone.
TOLERANCES = {
    "__default__": (0.0, 0.0),
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
    "features": "detect_features",
    "matches": "match_features",
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

    if case["kind"] in ("homography", "fundamental"):
        arrays = [imageio_utils.load(input_path(name))
                  for name in feature_cases.PAIR]
        estimate = (feature_runner.estimate_homography
                    if case["kind"] == "homography"
                    else feature_runner.estimate_fundamental)
        return [estimate(impl, case["config"], case["features"], arrays,
                         case["inlier_scale"])]

    if case["kind"] == "calibration":
        return [calib_runner.load_calibration(
                    os.path.join(REPO_ROOT, calib_cases.CALIBRATIONS[name]))
                for name in case["inputs"]]

    if case["kind"] == "nodes":
        return [calib_runner.dump_document(
                    os.path.join(REPO_ROOT, calib_cases.DOCUMENTS[name]))
                for name in case["inputs"]]

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

        for member in sorted(expected):
            got = np.asarray(actual[member])
            want = np.asarray(expected[member])

            assert got.shape == want.shape, (
                "{} {} '{}': shape {} != recorded {}".format(
                    case_id(item), name, member, got.shape, want.shape))

            if want.size == 0:
                continue

            difference = np.abs(got.astype(np.float64) -
                                want.astype(np.float64))
            assert difference.max() <= ARRAY_TOLERANCE, (
                "{} {} '{}': max difference {} exceeds {}".format(
                    case_id(item), name, member, difference.max(),
                    ARRAY_TOLERANCE))


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

    # Once the recorded name is an alias of the replacement, running it under
    # either name runs our code, so a documented divergence applies to both
    replacements = REPLACEMENTS_BY_GROUP.get(group, case_spec.REPLACEMENTS)
    replacing = ( impl != case["impl"] or case["impl"] in replacements )

    if impl != case["impl"] and not runner.is_registered( case["kind"], impl ):
        pytest.skip("{} is not registered in this build".format(impl))

    if replacing:
        whole_case = case_spec.divergence_reason(case["impl"], case["variant"])

        if whole_case:
            # Still run it, so a crash or a refused config is caught
            run_case(case, impl)
            pytest.skip("deliberate divergence: {}".format(whole_case))

    check_refusals(item, case, impl)

    outputs = run_case(case, impl)

    if case["kind"] == "detect":
        check_detections(item, case, outputs, group)
        return

    if case["kind"] in ("features", "matches", "homography", "fundamental"):
        check_array_case(item, case, outputs, group)
        return

    if case["kind"] in ("calibration", "nodes"):
        check_json_case(item, case, outputs, group)
        return

    max_tol, mean_tol = TOLERANCES.get(impl, TOLERANCES["__default__"])
    unstable = case.get("unstable", {})

    names = (list(case["outputs"]) if case["kind"] == "pipeline"
             else case["inputs"])

    for name, actual in zip(names, outputs):
        # A path the recording left uninitialised, which the replacement
        # deliberately implements properly instead. cases.py says why
        if replacing and case_spec.divergence_reason(
                case["impl"], case["variant"], name):
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
