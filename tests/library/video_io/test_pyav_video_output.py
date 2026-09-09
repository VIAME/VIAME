"""The PyAV writer, held to what a written video has to contain.

Read back with the PyAV reader, which `tests/golden/test_video.py` holds to
the C++ reader's recording, so a round trip that comes back right means both
ends agree with what the arrows did.

Run just these:  ctest -R "unit:video_io"
"""

import numpy as np
import pytest

# The C++ ffmpeg reader deadlocks inside open() if PyAV has driven libav
# earlier in the same process, so every video touched here goes through one
# writer and one reader and nothing else; see tests/golden/video_runner.py
# for the isolation the golden tests need for the same reason.

WIDTH, HEIGHT, FRAMES, RATE = 64, 48, 30, 10.0

# What the C++ writer registered, which a pipeline may set on this one
EXPECTED_KEYS = {
    "approximate", "bitrate", "codec_name", "crf", "cuda_device_index",
    "cuda_enabled", "format_name", "frame_rate_den", "frame_rate_num",
    "height", "pixel_format", "width",
}

# A lossy codec at its default quality, so a pixel is allowed to move. The
# figure the plan asks for; the measured mean is around 2.1
PIXEL_TOLERANCE = 3


@pytest.fixture(scope="module", autouse=True)
def modules():
    from kwiver.vital.modules import load_known_modules
    load_known_modules()


def writer(**config):
    from kwiver.vital.algo import VideoOutput

    algo = VideoOutput.create("pyav")

    if config:
        cfg = algo.get_configuration()
        for key, value in config.items():
            cfg.set_value(key, str(value))
        algo.set_configuration(cfg)

    return algo


def source_frames():
    """Frames with structure in all three channels and along time."""
    frames = []

    for index in range(FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = (np.arange(WIDTH)[None, :] + index * 4) % 256
        frame[:, :, 1] = (np.arange(HEIGHT)[:, None] * 3) % 256
        frame[:, :, 2] = (index * 8) % 256
        frames.append(frame)

    return frames


def write(path, frames, settings=True, **config):
    from kwiver.vital.types import Image, ImageContainer, Timestamp
    from kwiver.vital.types.video_settings import VideoSettings

    algo = writer(**config)
    algo.open(path,
              VideoSettings(WIDTH, HEIGHT, RATE) if settings else None)

    for index, frame in enumerate(frames):
        stamp = Timestamp()
        stamp.set_frame(index + 1)
        stamp.set_time_seconds(index / RATE)
        algo.add_image(ImageContainer(Image(frame)), stamp)

    algo.close()

    return algo


def read(path):
    from kwiver.vital.algo import VideoInput

    algo = VideoInput.create("pyav")
    algo.open(path)

    frames = []

    try:
        while algo.next_frame():
            frames.append(np.asarray(algo.frame_image().image().asarray()))
    finally:
        algo.close()

    return frames


def test_configuration_carries_the_keys_the_arrow_had():
    assert set(writer().get_configuration().available_values()) == EXPECTED_KEYS


def test_configuration_round_trips():
    algo = writer(codec_name="h264", bitrate="400000", pixel_format="yuv422p",
                  frame_rate_num="25", frame_rate_den="1")
    cfg = algo.get_configuration()

    assert cfg.get_value("codec_name") == "h264"
    assert cfg.get_value("bitrate") == "400000"
    assert cfg.get_value("pixel_format") == "yuv422p"
    assert cfg.get_value("frame_rate_num") == "25"


def test_round_trip_keeps_every_frame_and_its_pixels(tmp_path):
    path = str(tmp_path / "round_trip.mp4")
    frames = source_frames()

    write(path, frames)
    decoded = read(path)

    assert len(decoded) == FRAMES, \
        "wrote {} frames, read back {}".format(FRAMES, len(decoded))

    for frame in decoded:
        assert frame.shape == (HEIGHT, WIDTH, 3)

    difference = np.abs(np.asarray(decoded, dtype=np.int32) -
                        np.asarray(frames, dtype=np.int32))

    assert difference.mean() < PIXEL_TOLERANCE, (
        "mean absolute pixel difference {:.3f} over a round trip"
        .format(difference.mean()))


def test_frame_rate_comes_from_the_config_when_there_are_no_settings(tmp_path):
    """What a writer gets once nothing supplies concrete video settings."""
    import av

    path = str(tmp_path / "from_config.mp4")

    write(path, source_frames(), settings=False,
          width=WIDTH, height=HEIGHT, frame_rate_num=25, frame_rate_den=1)

    with av.open(path) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == 25
        assert stream.width == WIDTH
        assert stream.height == HEIGHT


def test_open_without_geometry_or_rate_is_an_error(tmp_path):
    algo = writer()

    with pytest.raises(Exception, match="width, height, and frame rate"):
        algo.open(str(tmp_path / "nothing.mp4"), None)

    assert not algo.good()


def test_codec_and_pixel_format_are_honoured(tmp_path):
    import av

    path = str(tmp_path / "h264.mp4")
    write(path, source_frames()[:5], codec_name="libx264",
          pixel_format="yuv444p")

    with av.open(path) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.pix_fmt == "yuv444p"


def test_close_is_idempotent_and_good_follows_it(tmp_path):
    path = str(tmp_path / "closed.mp4")
    algo = write(path, source_frames()[:5])

    assert not algo.good()
    algo.close()
    assert not algo.good()


def test_grayscale_input_is_written(tmp_path):
    """A single channel image, which the arrow mapped to gray rather than RGB."""
    path = str(tmp_path / "gray.mp4")
    frames = [frame[:, :, 0].copy() for frame in source_frames()[:10]]

    write(path, frames)
    decoded = read(path)

    assert len(decoded) == 10
