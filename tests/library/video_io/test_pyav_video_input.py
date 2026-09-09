"""What the PyAV reader guarantees beyond the golden replay.

`tests/golden/test_video.py` holds this reader to the C++ one's recording:
frame counts, numbering, timestamps, pixel digests and seek landings. What
is here is the behaviour the recording does not reach, because the recording
reads each clip straight through from the first frame.

Run just these:  ctest -R "unit:video_io"
"""

import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.abspath(os.path.join(HERE, "..", "..", "golden"))
CLIP = os.path.join(GOLDEN, "inputs", "clip.mp4")
MANIFEST = os.path.join(GOLDEN, "video", "manifest.json")


@pytest.fixture(scope="module", autouse=True)
def modules():
    from kwiver.vital.modules import load_known_modules
    load_known_modules()


def reader():
    from kwiver.vital.algo import VideoInput
    return VideoInput.create("pyav")


def recorded():
    with open(MANIFEST) as handle:
        return json.load(handle)["clips"]["clip.mp4"]


def test_seeking_first_keeps_the_timestamp_origin():
    """A seek before any frame is read must not become time zero.

    Times are relative to the start of the video. The origin was being taken
    from the first frame the reader happened to see, which after a seek is
    the frame seeked to, so every time afterwards was measured from there.
    """
    algo = reader()
    algo.open(CLIP)

    try:
        assert algo.seek_frame(15)
        assert algo.frame_timestamp().get_time_seconds() == pytest.approx(
            recorded()["timestamps"][14], abs=1e-6)
    finally:
        algo.close()


def test_seeking_after_reading_keeps_the_same_origin():
    algo = reader()
    algo.open(CLIP)

    try:
        assert algo.next_frame()
        assert algo.seek_frame(15)
        assert algo.frame_timestamp().get_time_seconds() == pytest.approx(
            recorded()["timestamps"][14], abs=1e-6)
    finally:
        algo.close()


def test_seek_time_lands_on_the_frame_covering_that_time():
    algo = reader()
    algo.open(CLIP)

    try:
        target = recorded()["timestamps"][9]
        assert algo.seek_time(int(round(target * 1000000)))
        assert algo.frame_timestamp().get_time_seconds() == pytest.approx(
            target, abs=1e-6)
    finally:
        algo.close()
