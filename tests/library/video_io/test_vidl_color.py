"""VIDL must reproduce the VXL recording, not the FFmpeg arrow's RGB values."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "golden"


@pytest.mark.parametrize("use_cli", [False, True])
def test_vidl_matches_original_vxl_pixels(use_cli):
    from viame.modules import load_known_modules
    from viame.algo import VideoInput
    load_known_modules()
    recording = json.loads((ROOT / "video/vidl_pixels.json").read_text())
    assert hashlib.sha256((ROOT / "inputs/clip.mp4").read_bytes()).hexdigest() == recording["clip_sha256"]
    expected = recording["frames"]
    reader = VideoInput.create("vidl_ffmpeg")
    config = reader.get_configuration()
    config.set_value("time_source", "start_at_0")
    config.set_value("use_cli", str(use_cli).lower())
    reader.set_configuration(config)
    reader.open(str(ROOT / "inputs/clip.mp4"))
    try:
        for record in expected:
            assert reader.next_frame()
            array = reader.frame_image().image().asarray()
            assert array.dtype == np.uint8
            assert hashlib.sha256(array.tobytes()).hexdigest()[:16] == record["sha256"]
    finally:
        reader.close()


def test_vidl_color_preserves_black_level_and_saturates():
    from viame.video_io.vidl_color import planar_yuv_to_rgb
    y = np.array([[16, 235], [0, 255]], dtype=np.uint8)
    neutral = np.full((1, 1), 128, dtype=np.uint8)
    np.testing.assert_array_equal(planar_yuv_to_rgb(y, neutral, neutral), np.repeat(y[:, :, None], 3, axis=2))
    result = planar_yuv_to_rgb(np.array([[0, 255]], np.uint8),
                               np.array([[0, 255]], np.uint8), np.array([[0, 255]], np.uint8))
    np.testing.assert_array_equal(result, [[[0, 136, 0], [255, 121, 255]]])


def test_vidl_timestamp_truncation_at_sampling_boundary():
    from fractions import Fraction
    from types import SimpleNamespace
    from viame.video_io.pyav_video_input import VidlFFmpegVideoInput
    reader = VidlFFmpegVideoInput()
    # Recorded from main: nominal 7.4 seconds is stored as 7,399,999 us.
    # Rounding this up changes the frame selected by the 5 Hz downsampler.
    frame = SimpleNamespace(pts=113664, time_base=Fraction(1, 15360))
    assert reader._vidl_time_usec(frame) == 7399999


@pytest.mark.parametrize("case", ["flush_clip", "full_range_clip"])
def test_vidl_decoder_flush_matches_main(case):
    from viame.modules import load_known_modules
    from viame.algo import VideoInput
    load_known_modules()
    recording = json.loads((ROOT / "video/vidl_pixels.json").read_text())[case]
    clip = ROOT / "inputs" / recording["clip"]
    assert hashlib.sha256(clip.read_bytes()).hexdigest() == recording["clip_sha256"]
    reader = VideoInput.create("vidl_ffmpeg")
    reader.open(str(clip))
    actual = []
    try:
        while reader.next_frame():
            stamp = reader.frame_timestamp()
            array = reader.frame_image().image().asarray()
            actual.append(dict(frame=stamp.get_frame(), time_usec=stamp.get_time_usec(),
                               sha256=hashlib.sha256(array.tobytes()).hexdigest()[:16]))
    finally:
        reader.close()
    assert actual == recording["frames"]
