# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Video reading on PyAV, replacing the C++ FFmpeg arrow.

PyAV is a thin binding over the same libraries the arrow used, so decoding
behaviour is unchanged; what moves is where the binding lives, from a
compiled dependency of the C++ build to a wheel from PyPI.

Behaviour held to `tests/golden/video/manifest.json`, recorded from the C++
reader before it was replaced:

* frames are numbered from one;
* the timestamp is the frame's presentation time relative to the first
  frame, rounded to whole microseconds, not its index divided by the frame
  rate, so a variable frame rate video keeps its real timing;
* an 8 bit source decodes to uint8 RGB and a deeper one to uint16 RGB;
* `num_frames` reports what the container claims, and counts frames only if
  it claims nothing.
"""

import logging

import numpy as np

from kwiver.vital.algo import VideoInput
from kwiver.vital.types import Image, ImageContainer, Timestamp

logger = logging.getLogger(__name__)

# Microseconds, which is the unit vital timestamps carry.
MICROSECONDS = 1000000


class PyAVVideoInput(VideoInput):
    """Read a video with PyAV."""

    def __init__(self):
        VideoInput.__init__(self)

        # Config, carrying the keys the C++ reader had so that a pipeline
        # setting one is not silently ignored. Several describe behaviour
        # PyAV gives us for free or that VIAME never used; those are accepted
        # and logged rather than pretended to.
        self._filter_desc = "yadif=deint=1"
        self._format_name = ""
        self._real_time = False
        self._use_misp_timestamps = False
        self._audio_enabled = True
        self._klv_enabled = True

        # From the reader this replaces the name of, whose keys the tooling
        # sets: see tests/baseline/pending.json
        self._start_at_frame = 0
        self._stop_after_frame = 0
        self._output_nth_frame = 1

        self._container = None
        self._stream = None
        self._frames = None
        self._filename = ""

        self._frame = None
        self._number = 0
        self._exhausted = False
        self._count = None
        self._start_ts = None
        self._graph = None
        self._graph_source = None
        self._graph_sink = None
        self._graph_deep = None

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
        self._use_misp_timestamps = _as_bool(cfg.get_value("use_misp_timestamps"))
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
        import av

        self.close()

        options = {"format": self._format_name} if self._format_name else {}
        self._container = av.open(video_name, **options)

        streams = self._container.streams.video

        if not streams:
            raise RuntimeError("no video stream in " + str(video_name))

        self._stream = streams[0]

        # Threaded decode is what makes this competitive with the C++ reader
        self._stream.thread_type = "AUTO"

        self._frames = self._container.decode(self._stream)
        self._filename = video_name
        self._frame = None
        self._number = 0
        self._exhausted = False
        self._start_ts = None

        declared = self._stream.frames
        self._count = int(declared) if declared else None

        self._build_graph()

    def close(self):
        if self._container is not None:
            self._container.close()

        self._container = None
        self._stream = None
        self._frames = None
        self._frame = None
        self._number = 0
        self._exhausted = False
        self._count = None
        self._start_ts = None
        self._graph = None
        self._graph_source = None
        self._graph_sink = None
        self._graph_deep = None

    # ------------------------------------------------------------------
    # Stepping

    def next_frame(self, timeout=0):
        if self._frames is None or self._exhausted:
            return False

        while True:
            frame = self._next_filtered()

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
        """Seek by decoding from the preceding keyframe.

        Seeking on the container lands on a keyframe, so the frames between
        it and the one asked for are decoded and dropped. That is what makes
        the frame actually returned the one requested.
        """
        if self._container is None or frame_number < 1:
            return False

        rate = self._average_rate()

        if not rate:
            return False

        target = int((frame_number - 1) / rate / self._stream.time_base)
        self._container.seek(target, stream=self._stream, backward=True)
        self._frames = self._container.decode(self._stream)
        self._exhausted = False

        # The seek lands at or before the target; step forward to it
        for frame in self._frames:
            number = self._number_of(frame, rate)

            if number >= frame_number:
                self._frame = frame
                self._number = number
                return True

        self._exhausted = True
        self._frame = None
        return False

    def seek_time(self, time_usec, timeout=0):
        if self._container is None:
            return False

        target = int(time_usec / MICROSECONDS / self._stream.time_base)
        self._container.seek(target, stream=self._stream, backward=True)
        self._frames = self._container.decode(self._stream)
        self._exhausted = False

        for frame in self._frames:
            if frame.pts is not None and \
                    frame.pts * self._stream.time_base * MICROSECONDS >= time_usec:
                self._frame = frame
                self._number = self._number_of(frame, self._average_rate())
                return True

        self._exhausted = True
        self._frame = None
        return False

    # ------------------------------------------------------------------
    # The current frame

    def frame_timestamp(self):
        stamp = Timestamp()

        if self._frame is None:
            return stamp

        stamp.set_frame(self._number)

        seconds = self._seconds_of(self._frame)

        if seconds is not None:
            stamp.set_time_seconds(seconds)

        return stamp

    def frame_image(self):
        if self._frame is None:
            return None

        return ImageContainer(Image(_planar_rgb(self._frame,
                                                self._graph_deep)))

    def _build_graph(self):
        """The filter chain every frame passes through.

        Frames go through libavfilter rather than `to_ndarray(format=...)`
        for two reasons. The configured `filter_desc` has to be applied, as
        the C++ reader applies it; and swscale's defaults differ from that
        reader by up to 3 counts on nearly every pixel, because they
        interpolate chroma rather than taking the nearest sample and write
        limited-range RGB. `SCALE_FLAGS` mirrors the flags in
        `arrows/ffmpeg/ffmpeg_convert_image.cxx`, and with them the result is
        bit identical.

        A filter such as the default `yadif` holds a frame back before it
        emits one, which is why stepping pushes and pulls rather than
        converting a frame in place.
        """
        import av

        deep = _bit_depth(self._stream.codec_context.format) > 8

        graph = av.filter.Graph()
        source = graph.add_buffer(template=self._stream)

        head = source

        if self._filter_desc.strip():
            head = _add_chain(graph, head, self._filter_desc)

        scale = graph.add("scale", SCALE_FLAGS)

        # Planar rather than interleaved RGB: vital images are planar, so
        # handing the interleaved layout to `Image` costs a strided per pixel
        # copy, which at 1080p is 14 ms a frame against 0.5 for a memcpy
        fmt = graph.add("format", "gbrp16le" if deep else "gbrp")
        sink = graph.add("buffersink")

        head.link_to(scale)
        scale.link_to(fmt)
        fmt.link_to(sink)
        graph.configure()

        self._graph = graph
        self._graph_source = source
        self._graph_sink = sink
        self._graph_deep = deep

    def _next_filtered(self):
        """The next frame out of the filter chain, or None at the end."""
        import av

        while True:
            try:
                return self._graph_sink.pull()
            except av.error.BlockingIOError:
                pass
            except av.error.EOFError:
                return None

            try:
                self._graph_source.push(next(self._frames))
            except StopIteration:
                # Flushing lets a filter emit whatever it was holding
                self._graph_source.push(None)
                self._frames = iter(())

    def frame_metadata(self):
        return []

    def frame_rate(self):
        return float(self._average_rate() or -1.0)

    def filename(self):
        return self._filename

    def end_of_video(self):
        return self._exhausted or self._frames is None

    def good(self):
        return self._frame is not None

    def num_frames(self):
        if self._count is not None:
            return self._count

        if self._container is None:
            return 0

        # The container did not say, so count without decoding
        self._count = sum(1 for _ in self._container.demux(self._stream)
                          if _.pts is not None)
        self._container.seek(0, stream=self._stream, backward=True)
        self._frames = self._container.decode(self._stream)
        return self._count

    # ------------------------------------------------------------------

    def _average_rate(self):
        if self._stream is None:
            return None

        rate = self._stream.average_rate or self._stream.guessed_rate
        return float(rate) if rate else None

    def _seconds_of(self, frame):
        """Presentation time relative to the first frame, in whole
        microseconds.

        The offset and the rounding both come from the C++ reader, which
        reports time from zero rather than from the container's own origin
        and stores whole microseconds.
        """
        if frame.pts is None:
            return None

        if self._start_ts is None:
            self._start_ts = frame.pts

        # The frame's own time base, not the stream's: filters may rescale it
        base = frame.time_base or self._stream.time_base

        offset = float((frame.pts - self._start_ts) * base)

        return int(offset * MICROSECONDS + 0.5) / MICROSECONDS

    def _number_of(self, frame, rate):
        """Frame number, from one, of a frame reached by seeking."""
        seconds = self._seconds_of(frame)

        if seconds is None or not rate:
            return self._number + 1

        return int(round(seconds * rate)) + 1


# The swscale setup arrows/ffmpeg used: nearest-neighbour chroma, accurate
# rounding, full chroma interpolation and input, and full-range RGB out.
SCALE_FLAGS = ("flags=full_chroma_int+full_chroma_inp+accurate_rnd+bitexact"
               "+neighbor:out_range=full")


def _add_chain(graph, head, description):
    """Append a comma separated filter description to the graph."""
    for step in description.split(","):
        step = step.strip()

        if not step:
            continue

        name, _, arguments = step.partition("=")
        node = graph.add(name, arguments or None)
        head.link_to(node)
        head = node

    return head


def _planar_rgb(frame, deep):
    """The frame as an (h, w, 3) array laid out the way vital stores images.

    The filter chain emits planar RGB, which ffmpeg orders G, B, R. Copying
    the three planes into one buffer in R, G, B order and then viewing it
    transposed gives an array whose strides are exactly vital's own
    (`w_step` 1, `h_step` width, `d_step` width * height), so constructing
    the image is a single memcpy rather than a per pixel walk. The planes
    carry row padding, hence the slice to the real width.
    """
    dtype = np.uint16 if deep else np.uint8
    height, width = frame.height, frame.width

    planes = []

    for plane in frame.planes:
        row = np.frombuffer(plane, dtype=dtype)
        stride = plane.line_size // dtype(0).itemsize
        planes.append(row.reshape(-1, stride)[:height, :width])

    green, blue, red = planes

    planar = np.empty((3, height, width), dtype=dtype)
    planar[0] = red
    planar[1] = green
    planar[2] = blue

    return planar.transpose(1, 2, 0)


def _bit_depth(pixel_format):
    try:
        return max(component.bits for component in pixel_format.components)
    except Exception:
        return 8


def _as_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        PyAVVideoInput, "pyav", "Read a video with PyAV")
