"""Run a shipped pipeline over the golden fixture frames.

Shared by the recorder and the golden test. The pipelines read
`input_list.txt` from the working directory and write their frames beside it,
so a fresh directory per run is all the isolation needed.
"""

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


def run(pipeline):
    """Run one pipeline; return {output name: array}, sorted by name.

    Outputs are whatever image files the pipeline wrote, which is what the
    recording compares. A pipeline that writes nothing is a failure: it means
    the wiring changed.
    """
    workdir = tempfile.mkdtemp(prefix="golden_pipe_")

    try:
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

        # Own process group so a timeout takes the whole pipeline with it
        process = subprocess.Popen(
            command, cwd=workdir, start_new_session=True, text=True,
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

        outputs = {}
        inputs = {os.path.basename(path) for path in listing}

        for name in sorted(os.listdir(workdir)):
            if name in inputs or name == "input_list.txt":
                continue
            if not name.lower().endswith((".png", ".tif", ".tiff", ".jpg")):
                continue

            outputs[name] = imageio_utils.load(os.path.join(workdir, name))

        if not outputs:
            raise AssertionError("{} wrote no images".format(pipeline))

        return outputs
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
