# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Video reading through the `ffmpeg` binary, for when PyAV is not there.

`pyav_video_input` is the reader; this is the fallback it falls back to,
selected either by naming `ffmpeg_cli` as the reader type or by setting
`use_cli` on the PyAV one, and used automatically when `import av` fails.
The binary comes from the `imageio-ffmpeg` wheel, so a build that has the
wheel has a working reader with no system FFmpeg and no compiled binding.
`VIAME_FFMPEG_EXE` overrides it, and without the wheel it falls back to
whatever is on PATH; either has to be FFmpeg 5.0 or newer, for `-fps_mode`.

It decodes by running

    ffmpeg -i <file> -vf <filters>,scale=<flags>,format=gbrp,showinfo
           -f rawvideo -pix_fmt gbrp -

and reading planes off the pipe. The filter chain and its swscale flags are
the ones `pyav_video_input` uses, so the pixels are identical: the recorded
digests in `tests/golden/video/manifest.json` match through this path too.

Timestamps come from `showinfo`, which reports each frame's presentation
time after the filters have run, and are computed from the integer pts and
the filter chain's time base exactly as the PyAV reader computes them.

Two things this cannot do as well as the PyAV reader:

* `num_frames` is the duration times the frame rate rather than what the
  container claims, because the banner does not report a frame count. On a
  variable frame rate video that is an estimate;
* stepping backwards means restarting the process, so a seek costs a
  process launch. It is exact -- `-ss` before `-i` with `-copyts` decodes to
  the frame asked for and keeps the original timestamps, which
  `tests/golden/test_video.py` checks alongside the PyAV reader -- but it is
  not cheap, and a pipeline that seeks per frame will feel it.
"""

import logging
import os
import queue
import re
import subprocess
import threading

import numpy as np

from kwiver.vital.algo import VideoInput
from kwiver.vital.types import Image, ImageContainer, Timestamp

logger = logging.getLogger(__name__)

MICROSECONDS = 1000000

# The swscale setup arrows/ffmpeg used, spelled for the command line. Same
# flags as `pyav_video_input.SCALE_FLAGS`.
SCALE_FLAGS = ("scale=flags=full_chroma_int+full_chroma_inp+accurate_rnd"
               "+bitexact+neighbor:out_range=full")

# showinfo prints one of these per frame, after the filters have run
FRAME_LINE = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:(\S+)")
CONFIG_LINE = re.compile(r"config in time_base:\s*(\d+)/(\d+),\s*"
                         r"frame_rate:\s*(\d+)/(\d+)")
SIZE = re.compile(r"\bs:(\d+)x(\d+)\b")
SOURCE_FORMAT = re.compile(r"\bfmt:(\S+)")
DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")

# How long to wait for the showinfo line belonging to a frame already read
# off the pipe. They are produced together, so this only ever expires when
# the process has died.
TIMESTAMP_TIMEOUT = 10.0

_BIT_DEPTHS = {}


class FFmpegCliVideoInput(VideoInput):
    """Read a video by driving the ffmpeg binary."""

    def __init__(self):
        VideoInput.__init__(self)

        # The same keys as `pyav_video_input`, so that a config written for
        # one reader works with the other
        self._filter_desc = "yadif=deint=1"
        self._format_name = ""
        self._real_time = False
        self._use_misp_timestamps = False
        self._audio_enabled = True
        self._klv_enabled = True
        self._start_at_frame = 0
        self._stop_after_frame = 0
        self._output_nth_frame = 1

        self._filename = ""
        self._process = None
        self._stderr = None
        self._times = None
        self._state = {}
        self._pending_pts = None

        self._width = 0
        self._height = 0
        self._deep = False
        self._rate = None
        self._duration = None

        self._frame = None
        self._pts = None
        self._number = 0
        self._exhausted = False
        self._start_pts = None
        self._count = None

    # ------------------------------------------------------------------
    # Configuration

    def get_configuration(self):
        cfg = super(VideoInput, self).get_configuration()
        cfg.set_value("filter_desc", str(self._filter_desc))
        cfg.set_value("format_name", str(self._format_name))
        cfg.set_value("real_time", str(bool(self._real_time)).lower())
        cfg.set_value("use_misp_timestamps",
                      str(bool(self._use_misp_timestamps)).lower())
        cfg.set_value("audio_enabled", str(bool(self._audio_enabled)).lower())
        cfg.set_value("klv_enabled", str(bool(self._klv_enabled)).lower())
        cfg.set_value("start_at_frame", str(int(self._start_at_frame)))
        cfg.set_value("stop_after_frame", str(int(self._stop_after_frame)))
        cfg.set_value("output_nth_frame", str(int(self._output_nth_frame)))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._filter_desc = str(cfg.get_value("filter_desc"))
        self._format_name = str(cfg.get_value("format_name"))
        self._real_time = _as_bool(cfg.get_value("real_time"))
        self._use_misp_timestamps = _as_bool(
            cfg.get_value("use_misp_timestamps"))
        self._audio_enabled = _as_bool(cfg.get_value("audio_enabled"))
        self._klv_enabled = _as_bool(cfg.get_value("klv_enabled"))
        self._start_at_frame = int(cfg.get_value("start_at_frame"))
        self._stop_after_frame = int(cfg.get_value("stop_after_frame"))
        self._output_nth_frame = int(cfg.get_value("output_nth_frame"))

        if self._use_misp_timestamps:
            logger.warning("use_misp_timestamps is not implemented; frame "
                           "presentation times are used instead")

    def check_configuration(self, cfg):
        if int(cfg.get_value("output_nth_frame", "1")) < 1:
            logger.error("output_nth_frame must be at least 1")
            return False
        return True

    # ------------------------------------------------------------------
    # Opening and closing

    def open(self, video_name):
        self.close()

        self._filename = video_name
        self._probe(video_name)
        self._launch()
        self._learn_origin()

    def close(self):
        self._stop()

        self._filename = ""
        self._width = self._height = 0
        self._deep = False
        self._rate = None
        self._duration = None
        self._count = None

    def _stop(self):
        """End the current decode, if there is one.

        In this order: ask ffmpeg to stop, close the frame pipe in case it
        was blocked writing to it, wait for it to go, and only then let the
        stderr reader finish on the end of its own pipe. Closing a pipe out
        from under a thread that is reading it is how this deadlocks.
        """
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()

            _close(self._process.stdout)

            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

            if self._stderr is not None:
                self._stderr.join(timeout=5)

            _close(self._process.stderr)

        self._process = None
        self._stderr = None
        self._times = None
        self._state = {}
        self._pending_pts = None
        self._frame = None
        self._pts = None
        self._number = 0
        self._exhausted = False
        self._start_pts = None

    # ------------------------------------------------------------------
    # Stepping

    def next_frame(self, timeout=0):
        if self._process is None or self._exhausted:
            return False

        while True:
            frame = self._read_frame()

            if frame is None:
                self._exhausted = True
                self._frame = None
                return False

            self._number += 1

            if self._stop_after_frame and self._number > self._stop_after_frame:
                self._exhausted = True
                self._frame = None
                return False

            if self._selected(self._number):
                self._frame = frame
                return True

    def _selected(self, number):
        if self._start_at_frame and number < self._start_at_frame:
            return False

        if self._output_nth_frame > 1:
            first = self._start_at_frame or 1
            if (number - first) % self._output_nth_frame != 0:
                return False

        return True

    def seek_frame(self, frame_number, timeout=0):
        """Restart the decode at the frame asked for.

        `-ss` before `-i` decodes and discards up to the given time rather
        than stopping at a keyframe, so this lands exactly; `-copyts` keeps
        the container's own timestamps, which the frame times are relative
        to.
        """
        if not self._filename or frame_number < 1 or not self._rate:
            return False

        if not self._restart((frame_number - 1) / self._rate):
            return False

        frame = self._read_frame()

        if frame is None:
            self._exhausted = True
            return False

        self._frame = frame
        self._number = frame_number
        return True

    def seek_time(self, time_usec, timeout=0):
        if not self._filename or not self._rate:
            return False

        seconds = time_usec / MICROSECONDS

        if not self._restart(seconds):
            return False

        frame = self._read_frame()

        if frame is None:
            self._exhausted = True
            return False

        self._frame = frame
        self._number = int(round(seconds * self._rate)) + 1
        return True

    def _restart(self, seconds):
        """Decode again from `seconds`, keeping the timestamp origin."""
        start = self._start_pts

        self._stop()
        self._launch(seconds)

        # The origin is the first frame of the video, not of this decode, so
        # a seek must not redefine it
        self._start_pts = start

        return self._process is not None

    # ------------------------------------------------------------------
    # The current frame

    def frame_timestamp(self):
        stamp = Timestamp()

        if self._frame is None:
            return stamp

        stamp.set_frame(self._number)

        seconds = self._seconds_of(self._pts)

        if seconds is not None:
            stamp.set_time_seconds(seconds)

        return stamp

    def frame_image(self):
        if self._frame is None:
            return None

        return ImageContainer(Image(self._frame))

    def frame_metadata(self):
        return []

    def frame_rate(self):
        return float(self._rate or -1.0)

    def filename(self):
        return self._filename

    def end_of_video(self):
        return self._exhausted or self._process is None

    def good(self):
        return self._frame is not None

    def num_frames(self):
        """The duration times the frame rate.

        The banner carries no frame count, so unlike the PyAV reader this is
        computed rather than read. Exact for constant frame rate video, which
        is what every shipped pipeline reads.
        """
        if self._count is not None:
            return self._count

        if self._duration is None or not self._rate:
            return 0

        self._count = int(round(self._duration * self._rate))
        return self._count

    # ------------------------------------------------------------------
    # Driving the binary

    def _probe(self, video_name):
        """Geometry, rate, duration and source depth, in one cheap decode."""
        result = _run(self._arguments(
            video_name, filters="showinfo", frames=1,
            output=("-fps_mode", "passthrough", "-f", "null")))

        frame = None

        for line in result.splitlines():
            if FRAME_LINE.search(line):
                frame = line
                break

        if frame is None:
            raise RuntimeError(
                "ffmpeg decoded no frames from {}".format(video_name))

        size = SIZE.search(frame)
        source = SOURCE_FORMAT.search(frame)

        if not size or not source:
            raise RuntimeError(
                "could not read the format of {} from ffmpeg".format(
                    video_name))

        self._width, self._height = int(size.group(1)), int(size.group(2))
        self._deep = _bit_depth(source.group(1)) > 8

        duration = DURATION.search(result)

        if duration:
            self._duration = (int(duration.group(1)) * 3600 +
                              int(duration.group(2)) * 60 +
                              float(duration.group(3)))

        config = CONFIG_LINE.search(result)

        if config:
            self._rate = int(config.group(3)) / int(config.group(4) or 1)

        # The time base is deliberately not taken from here: the probe runs a
        # bare showinfo, and a filter such as the default yadif changes the
        # time base of what reaches it. The decode reports its own.

    def _arguments(self, video_name, filters, frames=None, output=None,
                   seconds=None):
        args = [_executable(), "-nostdin", "-hide_banner"]

        if seconds:
            # Before -i, so ffmpeg seeks rather than decoding the whole
            # video and throwing frames away; -copyts because the frame
            # times are relative to the start of the video, not of the seek
            args += ["-ss", "{:.9f}".format(seconds), "-copyts"]

        if self._format_name:
            args += ["-f", self._format_name]

        args += ["-i", video_name, "-map", "0:v:0", "-vf", filters]

        if frames:
            args += ["-frames:v", str(frames)]

        args += list(output or
                     # passthrough because ffmpeg's default for a raw output
                     # is a constant frame rate, which duplicates frames to
                     # fill the gaps in a variable rate source: clip_vfr.mp4
                     # comes out as 34 frames rather than its own 30
                     ("-fps_mode", "passthrough",
                      "-f", "rawvideo", "-pix_fmt", self._pixel_format()))
        args.append("-")

        return args

    def _pixel_format(self):
        return "gbrp16le" if self._deep else "gbrp"

    def _filters(self):
        chain = [part for part in (self._filter_desc.strip(),) if part]
        chain += [SCALE_FLAGS, "format=" + self._pixel_format(), "showinfo"]
        return ",".join(chain)

    def _launch(self, seconds=None):
        self._process = subprocess.Popen(
            self._arguments(self._filename, self._filters(), seconds=seconds),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0)

        self._times = queue.Queue()
        self._state = {}
        self._stderr = threading.Thread(
            target=_watch,
            args=(self._process.stderr, self._times, self._state),
            daemon=True)
        self._stderr.start()

    def _read_frame(self):
        """The next frame off the pipe, or None once the video ends."""
        planes = self._read_planes()

        if planes is None:
            return None

        self._pts = self._next_pts()

        # gbrp is green, blue, red; vital wants red, green, blue in its own
        # planar layout, which is what makes `Image` a memcpy
        green, blue, red = planes
        planar = np.empty(planes.shape, dtype=planes.dtype)
        planar[0], planar[1], planar[2] = red, green, blue

        return planar.transpose(1, 2, 0)

    def _read_planes(self):
        dtype = np.uint16 if self._deep else np.uint8
        count = 3 * self._height * self._width
        wanted = count * dtype(0).itemsize

        buffer = bytearray(wanted)
        view = memoryview(buffer)
        filled = 0

        while filled < wanted:
            read = self._process.stdout.readinto(view[filled:])

            if not read:
                break

            filled += read

        if filled == 0:
            return None

        if filled < wanted:
            logger.error("ffmpeg gave %d bytes of a %d byte frame of %s",
                         filled, wanted, self._filename)
            return None

        return np.frombuffer(buffer, dtype=dtype).reshape(
            3, self._height, self._width)

    def _learn_origin(self):
        """Fix the timestamp origin at the first frame of the video.

        Times are reported relative to the start of the video, so seeking
        before any frame has been read must not make the frame it lands on
        time zero. showinfo has already printed the first frame's time by
        the time this runs, so it is read here and put back for the frame it
        belongs to.
        """
        pts = self._next_pts()

        self._pending_pts = pts

        if pts is not None:
            self._start_pts = pts

    def _next_pts(self):
        """The presentation time showinfo reported for the frame just read."""
        if self._pending_pts is not None:
            pts, self._pending_pts = self._pending_pts, None
            return pts

        try:
            return self._times.get(timeout=TIMESTAMP_TIMEOUT)
        except queue.Empty:
            logger.error("ffmpeg reported no time for frame %d of %s",
                         self._number + 1, self._filename)
            return None

    def _seconds_of(self, pts):
        """Presentation time from the first frame, in whole microseconds.

        The origin and the rounding are the C++ reader's, which reports time
        from zero rather than from the container's own start.
        """
        base = self._state.get("time_base")

        if pts is None or base is None:
            return None

        if self._start_pts is None:
            self._start_pts = pts

        numerator, denominator = base
        offset = (pts - self._start_pts) * numerator / denominator

        return int(offset * MICROSECONDS + 0.5) / MICROSECONDS


def _watch(stream, times, state):
    """Pull presentation times off showinfo's output as they are printed.

    On its own thread because the frames come down one pipe and their times
    down another; leaving stderr unread would fill its buffer and stall the
    decode. The time base arrives on showinfo's first line, before any
    frame, so a time read here is always paired with a base already set.
    """
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace")
            match = FRAME_LINE.search(line)

            if match:
                times.put(int(match.group(2)))
                continue

            config = CONFIG_LINE.search(line)

            if config:
                state["time_base"] = (int(config.group(1)),
                                      int(config.group(2)))
    except (ValueError, OSError):
        # The pipe was closed under us, which is how a decode is stopped
        pass


def _close(pipe):
    if pipe is None:
        return

    try:
        pipe.close()
    except Exception:
        pass


def _executable():
    """The ffmpeg binary, from the wheel if there is one."""
    override = os.environ.get("VIAME_FFMPEG_EXE")

    if override:
        return override

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        logger.warning("imageio-ffmpeg is not installed; falling back to "
                       "whatever ffmpeg is on PATH")
        return "ffmpeg"


def _run(arguments):
    """Run ffmpeg and return what it said, ignoring what it wrote."""
    result = subprocess.run(
        arguments, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    return result.stderr.decode("utf-8", "replace")


def _bit_depth(name):
    """Bits per component of a pixel format, asked of ffmpeg itself.

    Parsing the name would mean a table that goes stale; `-pix_fmts` prints
    the depths, and one call answers for every format.
    """
    if not _BIT_DEPTHS:
        for line in _pixel_formats().splitlines():
            fields = line.split()

            if len(fields) != 5 or not fields[2].isdigit():
                continue

            depth = fields[4].split("-")[0]

            if depth.isdigit():
                _BIT_DEPTHS[fields[1]] = int(depth)

    return _BIT_DEPTHS.get(name, 8)


def _pixel_formats():
    result = subprocess.run(
        [_executable(), "-hide_banner", "-pix_fmts"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    return result.stdout.decode("utf-8", "replace")


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        FFmpegCliVideoInput, "ffmpeg_cli",
        "Read a video by driving the ffmpeg binary, for when PyAV is absent")
