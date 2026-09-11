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


# ----------------------------------------------------------------------------
# Measurement from annotations
# ----------------------------------------------------------------------------

VIAME_CSV_HEADER = (
    "# 1: Detection or Track-id,  2: Video or Image Identifier,  "
    "3: Unique Frame Identifier,  4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y),  "
    "8: Detection or Length Confidence,  9: Target Length (0 or -1 if "
    "invalid),  10-11+: Repeated Species, Confidence Pairs or Attributes")

# The margin the generated boxes put around their two keypoints. Only the
# keypoints are measured; the box is there because a track needs one.
TRACK_BOX_MARGIN = 12


def _track_rows(truth, image_name, side):
    """A viame CSV for the head and tail keypoints of every target."""
    rows = [VIAME_CSV_HEADER]

    for index, target in enumerate(truth):
        (head_x, head_y), (tail_x, tail_y) = target[side]

        rows.append(
            "{},{},0,{:.6f},{:.6f},{:.6f},{:.6f},1,0,fish,1,"
            "(kp) head {:.6f} {:.6f},(kp) tail {:.6f} {:.6f}".format(
                index, image_name,
                min(head_x, tail_x) - TRACK_BOX_MARGIN,
                min(head_y, tail_y) - TRACK_BOX_MARGIN,
                max(head_x, tail_x) + TRACK_BOX_MARGIN,
                max(head_y, tail_y) + TRACK_BOX_MARGIN,
                head_x, head_y, tail_x, tail_y))

    return "\n".join(rows) + "\n"


def _parse_track_csv(path):
    """`{track id: {"bbox": [...], "attributes": {...}, "keypoints": {...}}}`."""
    out = {}

    with open(path) as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split(",")

            if len(fields) < 9:
                continue

            entry = {
                "bbox": [float(value) for value in fields[3:7]],
                "attributes": {},
                "keypoints": {},
            }

            for cell in fields[9:]:
                if cell.startswith("(atr) "):
                    parts = cell[6:].split()
                    if len(parts) == 2:
                        entry["attributes"][parts[0]] = parts[1]
                elif cell.startswith("(kp) "):
                    parts = cell[5:].split()
                    if len(parts) == 3:
                        entry["keypoints"][parts[0]] = (float(parts[1]),
                                                        float(parts[2]))

            out[int(fields[0])] = entry

    return out


def run_measurement_pipeline(pipeline, settings=(), paired=False):
    """Run the measurement pipeline over the synthetic scene.

    Returns `{key: array}`: the track ids that got a measurement, their
    lengths, midpoints, ranges and RMS errors, and the right keypoints the
    matcher found. Keyed by nothing else, so a port that measures a
    different subset of the tracks shows up as a different `track_ids`.
    """
    import measurement_fixtures

    truth = measurement_fixtures.measurement_truth()
    k_left, k_right, rotation, translation = \
        measurement_fixtures.measurement_rig()

    workdir = tempfile.mkdtemp(prefix="golden_measure_")

    try:
        left = _write_list(workdir, ["measure_left_00"], "cam1_images.txt")
        right = _write_list(workdir, ["measure_right_00"], "cam2_images.txt")

        with open(os.path.join(workdir, "tracks1.csv"), "w") as handle:
            handle.write(_track_rows(
                truth, os.path.basename(left[0]), "left"))

        # The **left** file name, even for the right camera's tracks. The
        # pipeline connects `downsampler.output_2` -- camera one's file name
        # -- to both readers, and `read_object_track` returns only the rows
        # whose identifier matches what it was handed, so a right track file
        # named after the right image reads as empty. Recorded as a finding
        # rather than worked around silently.
        with open(os.path.join(workdir, "tracks2.csv"), "w") as handle:
            handle.write(_track_rows(truth, os.path.basename(left[0]),
                                     "right")
                         if paired else VIAME_CSV_HEADER + "\n")

        np.savez(os.path.join(workdir, "calibration_matrices.npz"),
                 cameraMatrixL=k_left, cameraMatrixR=k_right,
                 distCoeffsL=np.zeros((1, 5)), distCoeffsR=np.zeros((1, 5)),
                 R=rotation, T=translation.reshape(3, 1))

        command = ["kwiver", "runner",
                   pipeline_runner.pipeline_path(pipeline),
                   "-s", "track_reader1:file_name=tracks1.csv",
                   "-s", "track_reader2:file_name=tracks2.csv",
                   "-s", "measurer:calibration_file=./calibration_matrices.npz"]

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

        measured = _parse_track_csv(
            os.path.join(workdir, "computed_tracks1.csv"))
        matched = _parse_track_csv(
            os.path.join(workdir, "computed_tracks2.csv"))

        ids = sorted(identifier for identifier, entry in measured.items()
                     if "length" in entry["attributes"])

        if not ids:
            raise AssertionError(
                "{} measured nothing".format(pipeline))

        def attribute(identifier, name):
            return float(measured[identifier]["attributes"][name])

        def keypoint(identifier, name):
            found = matched.get(identifier, {}).get("keypoints", {})
            return found.get(name, (float("nan"), float("nan")))

        return {
            "track_ids": np.asarray(ids, dtype=np.float64).reshape(-1, 1),
            "length": np.asarray(
                [attribute(i, "length") for i in ids]).reshape(-1, 1),
            "midpoint": np.asarray(
                [[attribute(i, "midpoint_x"), attribute(i, "midpoint_y"),
                  attribute(i, "midpoint_z")] for i in ids]),
            "range": np.asarray(
                [attribute(i, "midpoint_range") for i in ids]).reshape(-1, 1),
            "rms": np.asarray(
                [attribute(i, "stereo_rms") for i in ids]).reshape(-1, 1),
            "right_head": np.asarray([keypoint(i, "head") for i in ids]),
            "right_tail": np.asarray([keypoint(i, "tail") for i in ids]),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
