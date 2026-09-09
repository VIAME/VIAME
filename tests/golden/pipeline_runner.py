"""Run a shipped pipeline over the golden fixture frames.

Shared by the recorder and the golden test. The pipelines read
`input_list.txt` from the working directory and write their frames beside it,
so a fresh directory per run is all the isolation needed.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cases as case_spec           # noqa: E402
import imageio_utils                # noqa: E402

TIMEOUT = 900


def install_dir():
    install = os.environ.get("VIAME_INSTALL")

    if not install:
        raise RuntimeError(
            "VIAME_INSTALL is not set; source setup_viame.sh first")

    return install


def input_path(name):
    for ext in (".png", ".npz", ".npy"):
        candidate = os.path.join(HERE, "inputs", name + ext)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("no fixture named '{}'".format(name))


def sourced_environment():
    """The environment a pipeline gets when the install is sourced.

    Deliberately built from `setup_viame.sh` in a fresh shell rather than
    inherited: importing the kwiver python package rewrites LD_LIBRARY_PATH,
    and a golden that shells out must not depend on whether the test process
    happened to import it first.
    """
    script = os.path.join(install_dir(), "setup_viame.sh")

    result = subprocess.run(
        ["bash", "-c", 'source "{}" >/dev/null 2>&1 && env -0'.format(script)],
        stdout=subprocess.PIPE, check=True)

    environment = {}

    for entry in result.stdout.decode("utf-8", "replace").split("\0"):
        key, separator, value = entry.partition("=")
        if separator:
            environment[key] = value

    return environment


def _execute(pipeline, workdir, settings=()):
    """Run the pipeline in `workdir` over the fixture frames.

    Returns the names of the files it was given, so a caller can tell its
    inputs from what the pipeline wrote.
    """
    listing = []
    for name in case_spec.PIPELINE_INPUTS:
        source = input_path(name)
        target = os.path.join(workdir, os.path.basename(source))
        shutil.copyfile(source, target)
        listing.append(target)

    with open(os.path.join(workdir, "input_list.txt"), "w") as handle:
        handle.write("\n".join(listing) + "\n")

    command = ["kwiver", "runner",
               os.path.join(install_dir(), "configs", "pipelines", pipeline)]

    for setting in settings:
        command += ["-s", setting]

    # Own process group so a timeout takes the whole pipeline with it
    process = subprocess.Popen(
        command, cwd=workdir, start_new_session=True, text=True,
        env=sourced_environment(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        _, stderr = process.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        raise AssertionError(
            "{} did not finish within {}s".format(pipeline, TIMEOUT))

    if process.returncode != 0:
        raise AssertionError("{} exited {}:\n{}".format(
            pipeline, process.returncode, stderr[-4000:]))

    return {os.path.basename(path) for path in listing} | {"input_list.txt"}


def run(pipeline):
    """Run one pipeline; return {output name: array}, sorted by name.

    Outputs are whatever image files the pipeline wrote, which is what the
    recording compares. A pipeline that writes nothing is a failure: it means
    the wiring changed.
    """
    workdir = tempfile.mkdtemp(prefix="golden_pipe_")

    try:
        inputs = _execute(pipeline, workdir)

        outputs = {}

        for name in sorted(os.listdir(workdir)):
            if name in inputs:
                continue
            if not name.lower().endswith((".png", ".tif", ".tiff", ".jpg")):
                continue

            outputs[name] = imageio_utils.load(os.path.join(workdir, name))

        if not outputs:
            raise AssertionError("{} wrote no images".format(pipeline))

        return outputs
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_video(pipeline, settings=()):
    """Run a pipeline that writes a video; describe what it wrote.

    A frame of encoded video is not worth committing and would mostly be
    measuring the codec, so what comes back is what a viewer would see: how
    many frames the file yields, how long it claims to be, and the stream's
    geometry, codec and rate.
    """
    workdir = tempfile.mkdtemp(prefix="golden_video_")

    try:
        inputs = _execute(pipeline, workdir, settings)

        written = sorted(
            name for name in os.listdir(workdir)
            if name not in inputs and
            name.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".ts")))

        if not written:
            raise AssertionError("{} wrote no video".format(pipeline))

        if len(written) > 1:
            raise AssertionError(
                "{} wrote more than one video: {}".format(
                    pipeline, ", ".join(written)))

        return describe_video(os.path.join(workdir, written[0]), written[0])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def describe_video(path, name):
    """What a written video contains, read back with ffprobe.

    ffprobe rather than a reader from this build, so that the recording says
    what the file holds rather than what one of the implementations under
    test makes of it.
    """
    fields = ("codec_name,width,height,pix_fmt,avg_frame_rate,"
              "nb_read_frames,duration")

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=" + fields, "-of", "json", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=sourced_environment(), check=True)

    stream = json.loads(result.stdout.decode())["streams"][0]

    return {
        "file": name,
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pixel_format": stream["pix_fmt"],
        "frame_rate": stream["avg_frame_rate"],
        "frames": int(stream["nb_read_frames"]),
        "duration": round(float(stream["duration"]), 6),
    }
