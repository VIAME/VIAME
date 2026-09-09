"""The subprocess reader, and the fallback that reaches for it.

The three recorded clips are replayed against `ffmpeg_cli` by
`tests/golden/test_video.py`, which is where the frame counts, timestamps
and pixel digests are checked. What is here is the rest: that its config is
the PyAV reader's config, that frame selection behaves the same, and that a
build with no PyAV still reads video.

Run just these:  ctest -R "unit:video_io"
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.abspath(os.path.join(HERE, "..", "..", "golden"))
CLIP = os.path.join(GOLDEN, "inputs", "clip.mp4")
MANIFEST = os.path.join(GOLDEN, "video", "manifest.json")

sys.path.insert(0, GOLDEN)

import pipeline_runner                # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def modules():
    from kwiver.vital.modules import load_known_modules
    load_known_modules()


def reader(impl="ffmpeg_cli", **config):
    from kwiver.vital.algo import VideoInput

    algo = VideoInput.create(impl)

    if config:
        cfg = algo.get_configuration()
        for key, value in config.items():
            cfg.set_value(key, str(value))
        algo.set_configuration(cfg)

    return algo


def numbers(algo, path=CLIP):
    """The frame numbers a reader yields over the clip."""
    algo.open(path)

    found = []

    try:
        while algo.next_frame():
            found.append(algo.frame_timestamp().get_frame())
    finally:
        algo.close()

    return found


def recorded():
    with open(MANIFEST) as handle:
        return json.load(handle)["clips"]["clip.mp4"]


def test_configuration_matches_the_pyav_reader():
    """A config written for one reader has to work with the other."""
    cli = set(reader().get_configuration().available_values())
    pyav = set(reader("pyav").get_configuration().available_values())

    # `use_cli` selects between the two, so it belongs only to the one that
    # does the selecting
    assert pyav - cli == {"use_cli"}
    assert not cli - pyav


def test_reads_every_frame():
    assert numbers(reader()) == recorded()["frame_numbers"]


def test_start_at_frame_skips_the_ones_before_it():
    assert numbers(reader(start_at_frame=5)) == list(range(5, 31))


def test_stop_after_frame_ends_there():
    assert numbers(reader(stop_after_frame=4)) == [1, 2, 3, 4]


def test_output_nth_frame_thins_the_stream():
    assert numbers(reader(output_nth_frame=3)) == list(range(1, 31, 3))


def test_frame_selection_composes():
    assert numbers(reader(start_at_frame=2, stop_after_frame=8,
                          output_nth_frame=2)) == [2, 4, 6, 8]


def test_output_nth_frame_of_zero_is_rejected():
    algo = reader()
    cfg = algo.get_configuration()
    cfg.set_value("output_nth_frame", "0")

    assert not algo.check_configuration(cfg)


def test_frame_rate_and_count_come_back():
    algo = reader()
    algo.open(CLIP)

    try:
        assert algo.frame_rate() == pytest.approx(10.0)
        assert algo.num_frames() == recorded()["frames"]
    finally:
        algo.close()


def test_seeking_lands_on_the_frame_asked_for():
    """`-ss` before `-i` decodes to the frame rather than to a keyframe."""
    algo = reader()
    algo.open(CLIP)

    try:
        for target in (1, 7, 15, 29):
            assert algo.seek_frame(target)
            assert algo.frame_timestamp().get_frame() == target

            assert algo.next_frame()
            assert algo.frame_timestamp().get_frame() == target + 1
    finally:
        algo.close()


def test_seeking_keeps_the_timestamp_origin():
    """Times stay relative to the start of the video, not of the seek."""
    algo = reader()
    algo.open(CLIP)

    try:
        assert algo.seek_frame(15)
        assert algo.frame_timestamp().get_time_seconds() == pytest.approx(
            recorded()["timestamps"][14], abs=1e-6)
    finally:
        algo.close()


def test_use_cli_sends_the_pyav_reader_down_this_path():
    algo = reader("pyav", use_cli="true")
    assert numbers(algo) == recorded()["frame_numbers"]


# The plan's gate for this task is that video still reads with `av`
# uninstalled. Uninstalling it would break every other test in the run, so
# the import is blocked in a child process instead, which is the same thing
# from the reader's point of view.
BLOCK_AV = textwrap.dedent("""
    import importlib.abc, importlib.machinery, json, sys

    class NoAV(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "av" or name.startswith("av."):
                raise ImportError("av is not installed")
            return None

    sys.meta_path.insert(0, NoAV())

    from kwiver.vital.modules import load_known_modules
    from kwiver.vital.algo import VideoInput
    import numpy as np

    load_known_modules()

    reader = VideoInput.create("pyav")
    reader.open({clip!r})

    frames = 0
    shapes = set()

    while reader.next_frame():
        frames += 1
        shapes.add(np.asarray(reader.frame_image().image().asarray()).shape)

    reader.close()
    print("RESULT " + json.dumps(
        {{"frames": frames, "shape": sorted(shapes)[0]}}))
""")


def test_reads_video_with_pyav_unimportable():
    script = BLOCK_AV.format(clip=CLIP)

    # A fresh environment built from setup_viame.sh rather than the one this
    # process happens to have: importing the kwiver package rewrites
    # LD_LIBRARY_PATH, and a second install on the path loads its plugins too
    result = subprocess.run([sys.executable, "-c", script],
                            env=pipeline_runner.sourced_environment(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert result.returncode == 0, result.stderr.decode()[-3000:]

    line = [entry for entry in result.stdout.decode().splitlines()
            if entry.startswith("RESULT ")]

    assert line, result.stdout.decode()[-3000:]

    written = json.loads(line[0][len("RESULT "):])

    assert written["frames"] == recorded()["frames"]
    assert written["shape"] == recorded()["shape"]
