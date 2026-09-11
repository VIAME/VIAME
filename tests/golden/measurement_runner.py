"""Run the stereo pipelines over the golden fixtures.

`pipeline_runner` writes one image list and reads back the images a pipeline
wrote. A stereo pipeline needs two lists and writes calibration files rather
than images, so this is its counterpart -- the parts that are the same, the
sourced environment and the timeout, come from there.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import imageio_utils                # noqa: E402
import measurement_cases            # noqa: E402
import pipeline_runner              # noqa: E402


def _write_list(workdir, names, filename):
    """Copy the fixtures into `workdir` and list them, as the pipelines want."""
    paths = []

    for name in names:
        source = pipeline_runner.input_path(name)
        target = os.path.join(workdir, os.path.basename(source))
        shutil.copyfile(source, target)
        paths.append(target)

    with open(os.path.join(workdir, filename), "w") as handle:
        handle.write("\n".join(paths) + "\n")

    return paths


def run_stereo_pipeline(pipeline, left_names, right_names, settings=()):
    """Run a two-camera pipeline; return what it wrote, as parsed values.

    The calibration pipelines write `intrinsics.yml`, `extrinsics.yml` and a
    JSON summary, so what comes back is those parsed rather than their bytes:
    a recording of the bytes would be a recording of the writer, which
    `tests/library/file_io` already covers.
    """
    from viame.file_io import _opencv_yaml

    workdir = tempfile.mkdtemp(prefix="golden_stereo_")

    try:
        _write_list(workdir, left_names, "cam1_images.txt")
        _write_list(workdir, right_names, "cam2_images.txt")

        command = ["kwiver", "runner",
                   os.path.join(pipeline_runner.install_dir(), "configs",
                                "pipelines", pipeline)]

        for setting in settings:
            command += ["-s", setting]

        process = subprocess.Popen(
            command, cwd=workdir, start_new_session=True, text=True,
            env=pipeline_runner.sourced_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            _, stderr = process.communicate(timeout=pipeline_runner.TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            raise AssertionError("{} did not finish within {}s".format(
                pipeline, pipeline_runner.TIMEOUT))

        if process.returncode != 0:
            raise AssertionError("{} exited {}:\n{}".format(
                pipeline, process.returncode, stderr[-4000:]))

        outputs = {}

        for name in sorted(os.listdir(workdir)):
            if not name.endswith((".yml", ".yaml")):
                continue

            outputs[name] = _opencv_yaml.read(os.path.join(workdir, name))

        if not outputs:
            raise AssertionError(
                "{} wrote no calibration files".format(pipeline))

        return outputs
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def calibration_arrays(outputs):
    """The calibration a stereo pipeline wrote, as `{key: array}`.

    One flat mapping rather than a file-per-document, because which file a
    matrix lands in is the writer's business and what a caller wants is the
    rig.
    """
    arrays = {}

    for document in outputs.values():
        for key, value in document.items():
            if not isinstance(value, dict) or "data" not in value:
                continue

            arrays[key] = np.asarray(value["data"], dtype=np.float64).reshape(
                int(value["rows"]), int(value["cols"]))

    return arrays


def run_mono_pipeline(pipeline, names, settings=()):
    """Run a single camera calibration pipeline; return what it wrote.

    One image list rather than two, and a JSON summary beside the YAML. Both
    come back as one flat `{key: array}` so that a scalar the JSON carries
    and a matrix the YAML carries are checked the same way -- the JSON's
    `image_width` is the one the rectification downstream is built at, so it
    is as much part of the answer as `M1` is.
    """
    import json

    from viame.file_io import _opencv_yaml

    workdir = tempfile.mkdtemp(prefix="golden_mono_")

    try:
        _write_list(workdir, names, "input_list.txt")

        command = ["kwiver", "runner",
                   pipeline_runner.pipeline_path(pipeline)]

        for setting in settings:
            command += ["-s", setting]

        process = subprocess.Popen(
            command, cwd=workdir, start_new_session=True, text=True,
            env=pipeline_runner.sourced_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            _, stderr = process.communicate(timeout=pipeline_runner.TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            raise AssertionError("{} did not finish within {}s".format(
                pipeline, pipeline_runner.TIMEOUT))

        if process.returncode != 0:
            raise AssertionError("{} exited {}:\n{}".format(
                pipeline, process.returncode, stderr[-4000:]))

        arrays = {}

        for name in sorted(os.listdir(workdir)):
            path = os.path.join(workdir, name)

            if name.endswith((".yml", ".yaml")):
                arrays.update(calibration_arrays(
                    {name: _opencv_yaml.read(path)}))
            elif name.endswith(".json"):
                with open(path) as handle:
                    summary = json.load(handle)

                for key, value in sorted(summary.items()):
                    arrays[key] = np.asarray([[float(value)]])

        if not arrays:
            raise AssertionError(
                "{} wrote no calibration".format(pipeline))

        return arrays
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
