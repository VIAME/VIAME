"""The script-facing frame reader, against the algorithm beside it.

`frames.read_frames` exists so a tool can read a video without loading the
plugin framework -- `tools/calibrate.py` used `cv2.VideoCapture` for this. The
contract worth holding is not "it decodes something" but that it decodes the
**same pixels** the `video_input:pyav` algorithm does, because a calibration
read from a video has to agree with a pipeline read from the same video.

Run just these:  ctest -R "unit:video_io:frames"
"""

import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.abspath(os.path.join(HERE, "..", "..", "..", "tests", "reference"))
CLIP = os.path.join(GOLDEN, "inputs", "clip.mp4")
VFR = os.path.join(GOLDEN, "inputs", "clip_vfr.mp4")
NOT_A_VIDEO = os.path.join(GOLDEN, "inputs", "bayer_bg.png")


@pytest.fixture(scope="module", autouse=True)
def modules():
    from viame.modules import load_known_modules
    load_known_modules()


def read(path):
    from viame.video_io import frames
    return list(frames.read_frames(path))


def test_frames_are_rgb_arrays_numbered_from_one():
    pairs = read(CLIP)

    assert len(pairs) > 0
    assert [number for _, number in pairs] == list(range(1, len(pairs) + 1))

    for array, _ in pairs:
        assert array.ndim == 3 and array.shape[2] == 3
        assert array.dtype == np.uint8
        assert array.flags["C_CONTIGUOUS"]

    # Freshly owned, so a caller may write into what it was handed.
    first = pairs[0][0]
    first[0, 0] = [1, 2, 3]
    assert list(pairs[0][0][0, 0]) == [1, 2, 3]


def test_frames_match_the_pipeline_reader_exactly():
    """The whole reason this module has its own scale flags.

    swscale's defaults interpolate chroma and write limited-range RGB, which
    differs from the C++ reader -- and from `cv2.VideoCapture` -- by up to
    three counts on nearly every pixel. `frames` uses `pyav_video_input`'s
    flags instead, so the two agree bit for bit.
    """
    from viame.algo import VideoInput

    algorithm = VideoInput.create("pyav")
    assert algorithm is not None
    algorithm.open(CLIP)
    try:
        expected = []
        while algorithm.next_frame(0):
            expected.append(np.asarray(algorithm.frame_image().asarray()))
    finally:
        algorithm.close()

    actual = [array for array, _ in read(CLIP)]

    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_count_frames_reports_what_the_container_claims():
    from viame.video_io import frames

    claimed = frames.count_frames(CLIP)
    assert claimed == len(read(CLIP))

    # A variable frame rate clip still has a frame count in its container.
    assert frames.count_frames(VFR) > 0


def test_a_file_with_no_video_stream_is_refused():
    from viame.video_io import frames

    with pytest.raises(Exception):
        frames.count_frames(os.path.join(GOLDEN, "inputs", "does_not_exist.mp4"))

    # A still image opens as a container with one video stream, so it decodes
    # rather than raising -- which is worth knowing, since a caller choosing
    # between this and an image reader cannot use "it failed" to tell them
    # apart.
    assert len(read(NOT_A_VIDEO)) >= 1


def test_filter_errors_propagate(monkeypatch):
    import av
    from viame.video_io import frames
    original = av.filter.Graph

    class BrokenGraph:
        def __init__(self):
            self.inner = original()
        def __getattr__(self,name):
            return getattr(self.inner,name)
        def pull(self):
            raise av.error.InvalidDataError(1094995529, "injected filter failure")

    monkeypatch.setattr(av.filter,"Graph",BrokenGraph)
    with pytest.raises(av.error.InvalidDataError,match="injected filter failure"):
        list(frames.read_frames(CLIP))
