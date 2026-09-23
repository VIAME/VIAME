"""A video has to survive a whole pipeline, not just the reader API.

`test_video.py` drives `VideoInput.create()` directly, which is how the
readers were verified when they were ported to python -- and it is why two
faults that broke *every* video pipeline went unnoticed for as long as they
did. Neither is reachable through the reader API:

  * `video_input_process` refuses a reader that does not advertise
    HAS_FRAME_DATA, and a python reader had no way to advertise anything
    because `set_capability` was protected and unbound.
  * `filename()` returned the container path, which `image_writer_process`
    writes to verbatim, so a frame was saved as `clip.mp4`.

So this case runs the real thing: a pipeline, over an mp4, writing frames.

Run just this:  ctest -R "pipeline:video"
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pipeline_runner              # noqa: E402

CLIP = os.path.join(HERE, "inputs", "clip.mp4")

# What `clip.mp4` holds: 30 frames at 10 fps. Asserted rather than counted so
# a reader that silently drops or doubles frames fails here.
EXPECTED_FRAMES = 30


@pytest.mark.parametrize("reader", ["vidl_ffmpeg", "ffmpeg", "pyav"])
def test_a_video_runs_through_a_pipeline(tmp_path, reader):
    """Every frame comes out, named by `file_name_template`."""
    assert os.path.exists(CLIP), CLIP

    result = subprocess.run(
        ["kwiver", "runner", pipeline_runner.pipeline_path("filter_enhance.pipe"),
         "-s", "input:video_filename=" + CLIP,
         "-s", "input:video_reader:type=" + reader],
        cwd=str(tmp_path), env=pipeline_runner.sourced_environment(),
        capture_output=True, text=True, timeout=600)

    written = sorted(p.name for p in tmp_path.glob("frame*.png"))

    assert written, (
        "no frames written with reader {!r}; the pipeline failed.\n"
        "This is what a reader that does not declare HAS_FRAME_DATA, or one "
        "that returns the container from filename(), looks like.\n{}".format(
            reader, result.stderr[-2000:]))

    assert len(written) == EXPECTED_FRAMES, (
        "{} frames from {!r}, expected {}".format(
            len(written), reader, EXPECTED_FRAMES))

    # Named from the template, not from the video: `clip.mp4` here means
    # `filename()` handed the writer the container again. Numbering starts at
    # one, as the frame numbers do, and matches the reference build of main.
    assert written[0] == "frame000001.png", written[:3]
    assert written[-1] == "frame%06d.png" % EXPECTED_FRAMES, written[-3:]
    assert not list(tmp_path.glob("*.mp4*")), "wrote output named after the video"
