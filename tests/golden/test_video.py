"""Hold the video reader and writer to what the C++ ones did before they were
replaced.

`tests/golden/video/manifest.json` was recorded from `arrows/ffmpeg` while it
was still the only reader. Every replacement has to reproduce it: the same
frames, the same presentation times, the same pixels for the frames that were
digested, and seeking that lands where it says it does.

Run just these:  ctest -L GOLDEN
Re-record:       see tests/golden/README.md
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pipeline_runner              # noqa: E402
import video_cases                  # noqa: E402
import video_runner                 # noqa: E402

MANIFEST = os.path.join(HERE, "video", "manifest.json")
INPUTS = os.path.join(HERE, "inputs")

# Implementations that have to reproduce the recording. Since P4-T05 every
# one of these is python: `ffmpeg` and `vidl_ffmpeg` are the names the C++
# arrow and the VXL reader answered to, and they are now aliases of the PyAV
# reader, which is the point of holding all of them to the C++ recording.
IMPLEMENTATIONS = ("ffmpeg", "vidl_ffmpeg", "pyav", "ffmpeg_cli")

# A presentation time is a rational converted to seconds, so it is compared
# with the microsecond tolerance the plan asks for rather than exactly.
TIME_TOLERANCE = 1e-6

# Where a replacement cannot reach that, with the reason and the measured
# bound. The C++ reader timed frames by libavcodec's `best_effort_timestamp`,
# which applies a heuristic that can correct an irregular presentation time by
# one tick; PyAV does not expose it, so the raw `pts` is used instead. The two
# agree except on frames the heuristic corrects, which in a variable frame
# rate clip means a difference of at most one tick of the container time base.
# Constant rate video is unaffected, and every shipped pipeline reads either
# an image list or constant rate video.
# Keyed by (implementation, clip). The CLI reader reads its times from
# libavfilter's showinfo, which reports what the same heuristic produced, so
# it matches exactly and gets no allowance.
_VFR = (
    # One container tick, plus the microsecond the recorded times are
    # rounded to
    1.0 / 10240 + 1e-6,
    "pts rather than best_effort_timestamp; PyAV does not expose the "
    "latter, and the two differ by at most one container tick on the "
    "frames the heuristic corrects",
)

TIMESTAMP_DIVERGENCE = {
    ("pyav", "clip_vfr.mp4"): _VFR,
    ("ffmpeg", "clip_vfr.mp4"): _VFR,
    ("vidl_ffmpeg", "clip_vfr.mp4"): _VFR,
}

# The replacement has to stay within this fraction of the recorded decode
# throughput. Timing is noisy on a shared machine, so this is a floor on a
# best-of-three, not a benchmark.
THROUGHPUT_FRACTION = 0.6


def manifest():
    if not os.path.exists(MANIFEST):
        pytest.skip("no video recording")

    with open(MANIFEST) as handle:
        return json.load(handle)


def cases():
    if not os.path.exists(MANIFEST):
        return []

    with open(MANIFEST) as handle:
        clips = sorted(json.load(handle)["clips"])

    return [(impl, clip) for impl in IMPLEMENTATIONS for clip in clips]


@pytest.fixture(scope="session", autouse=True)
def modules():
    video_runner.load_modules()


def case_id(case):
    return "{}:{}".format(*case)


@pytest.mark.parametrize("case", cases(), ids=case_id)
def test_reader_matches_recording(case):
    impl, clip = case

    if not video_runner.is_registered(impl):
        pytest.skip("{} is not registered in this build".format(impl))

    expected = manifest()["clips"][clip]
    frames = video_runner.run_isolated(impl, {}, os.path.join(INPUTS, clip))

    assert len(frames) == expected["frames"], (
        "{} yielded {} frames, recorded {}".format(
            impl, len(frames), expected["frames"]))

    assert [frame["frame"] for frame in frames] == expected["frame_numbers"], \
        "{} numbered its frames differently".format(impl)

    assert frames[0]["shape"] == expected["shape"]
    assert frames[0]["dtype"] == expected["dtype"], (
        "{} decoded {} as {}, recorded {}".format(
            impl, clip, frames[0]["dtype"], expected["dtype"]))

    tolerance = TIME_TOLERANCE
    reason = ""

    if (impl, clip) in TIMESTAMP_DIVERGENCE:
        tolerance, reason = TIMESTAMP_DIVERGENCE[(impl, clip)]

    for index, (actual, want) in enumerate(
            zip(frames, expected["timestamps"])):
        if want is None:
            continue
        assert actual["time"] is not None, \
            "{} frame {} has no time".format(impl, index + 1)
        assert abs(actual["time"] - want) <= tolerance, (
            "{} frame {} at {} s, recorded {} s{}".format(
                impl, index + 1, actual["time"], want,
                "; allowed " + reason if reason else ""))

    for number, want in sorted(expected["digests"].items()):
        assert frames[int(number) - 1]["sha256"] == want, (
            "{} decoded frame {} of {} differently".format(
                impl, number, clip))


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_seek_lands_on_the_frame_asked_for(impl):
    clip = os.path.join(INPUTS, "clip.mp4")
    landings = video_runner.run_isolated(impl, {}, clip, action="seek_probe")

    if landings is None:
        pytest.skip("{} is not registered in this build".format(impl))

    for target, landed, following in landings:
        assert landed == target, (
            "{} seeking to {} landed on {}".format(impl, target, landed))

        # And the frame after a seek is the next one, not a repeat
        assert following == target + 1, (
            "{} did not continue from {}".format(impl, target))


def test_replacement_keeps_up():
    """The replacement decodes at a usable fraction of the recorded rate."""
    recorded = manifest().get("throughput", {}).get("frames_per_second")

    if not recorded:
        pytest.skip("no throughput recorded")

    clip = os.path.join(INPUTS, "clip.mp4")
    best = video_runner.run_isolated("pyav", {}, clip, action="throughput")

    # The recorded figure is 1080p and this clip is smaller, so this only
    # catches a replacement that is slow in absolute terms rather than
    # comparing like with like; the like-for-like number is in STATUS.md
    assert best >= recorded * THROUGHPUT_FRACTION, (
        "pyav decoded at {:.1f} fps, under {:.0%} of the recorded {:.1f}"
        .format(best, THROUGHPUT_FRACTION, recorded))


# --------------------------------------------------------------------------
# The writer

# What a replacement writer is allowed to differ by, with the reason.
#
# The C++ writer never set a duration on the packets it muxed, so the mp4
# muxer derived each sample's duration from the next sample's decode time and
# gave the last one a duration of zero. The track then ends one frame before
# its final sample, and every decoder trims that sample: a pipeline that
# writes N frames produces a file that plays N-1. PyAV sets the duration, so
# the file holds what was written -- one more frame, and one frame longer.
#
# This is a fix, not a divergence to preserve, so it is spelled out per field
# rather than widened into a tolerance.
_TRIMMED = ("the C++ writer muxed packets with no duration, so the mp4 muxer "
            "trimmed the final frame; this one writes it")

WRITER_DIVERGENCE = {
    "pyav": _TRIMMED,
    "ffmpeg": _TRIMMED,
}


def writer_cases():
    if not os.path.exists(MANIFEST):
        return []

    with open(MANIFEST) as handle:
        recorded = json.load(handle).get("pipelines", {})

    return [(writer, pipeline)
            for writer in sorted(video_cases.VIDEO_WRITERS)
            for pipeline in sorted(recorded)]


@pytest.mark.parametrize("case", writer_cases(), ids=case_id)
def test_writer_matches_recording(case):
    writer, pipeline = case

    if not video_runner.is_registered(writer, interface="video_output"):
        pytest.skip("{} is not registered in this build".format(writer))

    expected = manifest()["pipelines"][pipeline]
    written = pipeline_runner.run_video(
        pipeline, video_cases.VIDEO_WRITERS[writer])

    # Geometry, codec and pixel format are the writer reproducing the stream
    # it was asked for, and no replacement has a reason to change them
    for field in ("codec", "width", "height", "pixel_format"):
        assert written[field] == expected[field], (
            "{} wrote {} {}, recorded {}".format(
                writer, field, written[field], expected[field]))

    frames, duration = expected["frames"], expected["duration"]
    reason = ""

    if writer in WRITER_DIVERGENCE:
        reason = "; allowed: " + WRITER_DIVERGENCE[writer]
        frames = frames + 1
        duration = pytest.approx(duration * frames / (frames - 1), rel=1e-3)

    assert written["frames"] == frames, (
        "{} wrote {} frames of {}, recorded {}{}".format(
            writer, written["frames"], pipeline, expected["frames"], reason))

    assert written["duration"] == duration, (
        "{} wrote {} s of {}, recorded {} s{}".format(
            writer, written["duration"], pipeline, expected["duration"],
            reason))
