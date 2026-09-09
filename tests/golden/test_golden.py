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

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cases as case_spec           # noqa: E402
import imageio_utils                # noqa: E402
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


def removed_names():
    """Names deliberately removed, from the baseline's removed.json.

    A recording of a name that is gone on purpose is history, not a contract.
    Skipping here rather than failing keeps the recording available for
    comparison if the decision is ever revisited.
    """
    if not os.path.exists(REMOVED_PATH):
        return {}

    with open(REMOVED_PATH) as handle:
        return {entry["name"]: entry.get("reason", "")
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


def collect_cases():
    cases = []

    for group in sorted(os.listdir(HERE)):
        manifest = load_manifest(group)

        if manifest is None:
            continue

        for case in manifest["cases"]:
            cases.append((group, case, case["impl"]))

            replacement = case_spec.REPLACEMENTS.get(case["impl"])
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


def run_case(case, impl):
    if case["kind"] == "image_filter":
        arrays = [imageio_utils.load(input_path(name)) for name in case["inputs"]]
        return runner.run_image_filter(impl, case["config"], arrays)

    if case["kind"] == "image_io":
        paths = [input_path(name) for name in case["inputs"]]
        return runner.run_image_io_load(impl, case["config"], paths)

    if case["kind"] == "pipeline":
        outputs = pipeline_runner.run(impl)
        missing = sorted(set(case["outputs"]) - set(outputs))
        assert not missing, "{} no longer wrote: {}".format(
            case["impl"], ", ".join(missing))
        return [outputs[name] for name in case["outputs"]]

    raise AssertionError("unknown case kind '{}'".format(case["kind"]))


REMOVED = removed_names()


@pytest.mark.parametrize("item", collect_cases(), ids=case_id)
def test_golden(item):
    group, case, impl = item

    if case["impl"] in REMOVED:
        pytest.skip("{} was removed on purpose: {}".format(
            case["impl"], REMOVED[case["impl"]]))

    # Once the recorded name is an alias of the replacement, running it under
    # either name runs our code, so a documented divergence applies to both
    replacing = ( impl != case["impl"] or
                  case["impl"] in case_spec.REPLACEMENTS )

    if impl != case["impl"] and not runner.is_registered( case["kind"], impl ):
        pytest.skip("{} is not registered in this build".format(impl))

    if replacing:
        whole_case = case_spec.divergence_reason(case["impl"], case["variant"])

        if whole_case:
            # Still run it, so a crash or a refused config is caught
            run_case(case, impl)
            pytest.skip("deliberate divergence: {}".format(whole_case))

    outputs = run_case(case, impl)

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
