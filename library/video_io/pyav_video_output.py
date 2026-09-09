# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Video writing on PyAV, replacing the C++ FFmpeg arrow.

The counterpart of `pyav_video_input`, and like it a thin binding over the
same libraries the arrow used, so what changes is where the binding lives
rather than how anything is encoded.

Behaviour held to the C++ writer:

* geometry and frame rate come from the settings `open` is given, with the
  config as the backup, and it is an error for any of the three to be
  missing by the time the stream is created;
* the codec is the configured one if there is one, else H.265 and then
  H.264, which is what the arrow's priority order picks for the containers
  VIAME writes;
* pixels reach the encoder through swscale with the arrow's exact flags, so
  a frame written here and a frame written there are the same bytes;
* presentation times are the frame index, so a video is written at the
  stream's rate whatever the timestamps say.

The one thing a writer must not get wrong is finishing: an encoder holds
frames back, and a container needs its trailer, so a file closed without
flushing is short and usually unplayable. `close` does both and is safe to
call more than once; `video_output_process::_finalize` is what calls it at
the end of a pipeline.
"""

import fractions
import logging

import numpy as np

from kwiver.vital.algo import VideoOutput

logger = logging.getLogger(__name__)

# What the arrow's codec priority comes to once the settings carry no codec
# of their own, which is the case for every VIAME pipeline: H.265 first, then
# H.264. `lite-removals.md` section 3.2 gives the default as h264; the arrow
# ranked H.265 above it and mp4 takes both, so that is what a video written
# today is, and that is what this reproduces.
PREFERRED_CODECS = ("hevc", "h264")

# swscale flags copied from `arrows/ffmpeg/ffmpeg_convert_image.cxx`. The
# defaults differ visibly: they interpolate chroma rather than taking the
# nearest sample, which moves nearly every pixel of a subsampled format.
# `in_range=full` because the incoming RGB is full range, `out_range` from
# the target format.
SCALE_FLAGS = ("flags=neighbor+accurate_rnd+bitexact+full_chroma_int"
               "+full_chroma_inp")

# Formats whose conventional range is full rather than limited, matching
# `color_range_from_pix_fmt` in the arrow. Everything YUV is limited.
FULL_RANGE_FORMATS = ("gray", "gray16le", "gbrp", "gbrp16le", "rgb24",
                      "rgb48le", "rgba", "bgr24", "yuvj420p", "yuvj422p",
                      "yuvj444p", "yuvj440p", "yuvj411p")


class PyAVVideoOutput(VideoOutput):
    """Write a video with PyAV."""

    def __init__(self):
        VideoOutput.__init__(self)

        # The keys the C++ writer registered, so a pipeline setting one is
        # not silently ignored. `cuda_enabled`, `cuda_device_index` and
        # `approximate` describe the arrow's hardware and fast paths, which
        # have no PyAV equivalent; they are accepted and logged.
        self._width = 0
        self._height = 0
        self._frame_rate_num = 0
        self._frame_rate_den = 1
        self._bitrate = 0
        self._crf = ""
        self._codec_name = ""
        self._cuda_enabled = False
        self._cuda_device_index = 0
        self._approximate = False
        self._format_name = ""
        self._pixel_format = "yuv420p"

        self._container = None
        self._stream = None
        self._graph = None
        self._graph_source = None
        self._graph_sink = None
        self._source_format = None
        self._filename = ""
        self._count = 0
        self._time_base = None

    # ------------------------------------------------------------------
    # Configuration

    def get_configuration(self):
        cfg = super(VideoOutput, self).get_configuration()
        cfg.set_value("width", str(int(self._width)))
        cfg.set_value("height", str(int(self._height)))
        cfg.set_value("frame_rate_num", str(int(self._frame_rate_num)))
        cfg.set_value("frame_rate_den", str(int(self._frame_rate_den)))
        cfg.set_value("bitrate", str(int(self._bitrate)))
        cfg.set_value("crf", str(self._crf))
        cfg.set_value("codec_name", str(self._codec_name))
        cfg.set_value("cuda_enabled", str(bool(self._cuda_enabled)).lower())
        cfg.set_value("cuda_device_index", str(int(self._cuda_device_index)))
        cfg.set_value("approximate", str(bool(self._approximate)).lower())
        cfg.set_value("format_name", str(self._format_name))
        cfg.set_value("pixel_format", str(self._pixel_format))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._width = int(cfg.get_value("width"))
        self._height = int(cfg.get_value("height"))
        self._frame_rate_num = int(cfg.get_value("frame_rate_num"))
        self._frame_rate_den = int(cfg.get_value("frame_rate_den"))
        self._bitrate = int(cfg.get_value("bitrate"))
        self._crf = str(cfg.get_value("crf"))
        self._codec_name = str(cfg.get_value("codec_name"))
        self._cuda_enabled = _as_bool(cfg.get_value("cuda_enabled"))
        self._cuda_device_index = int(cfg.get_value("cuda_device_index"))
        self._approximate = _as_bool(cfg.get_value("approximate"))
        self._format_name = str(cfg.get_value("format_name"))
        self._pixel_format = str(cfg.get_value("pixel_format"))

        if self._cuda_enabled:
            logger.warning(
                "cuda_enabled is set, but this writer encodes on the CPU; "
                "name a hardware encoder in codec_name instead")

    def check_configuration(self, cfg):
        # The C++ writer accepted anything here and failed in open() instead;
        # the one value that cannot be recovered from is a zero denominator,
        # which would make the frame rate undefined rather than merely absent
        if int(cfg.get_value("frame_rate_den", "1") or 1) == 0:
            logger.error("frame_rate_den must not be zero")
            return False
        return True

    # ------------------------------------------------------------------
    # Opening and closing

    def open(self, video_name, settings=None):
        import av

        self.close()

        width, height, rate = self._resolve(settings)

        options = {}
        if self._format_name:
            options["format"] = self._format_name

        self._container = av.open(video_name, mode="w", **options)
        self._filename = video_name
        self._count = 0

        try:
            self._stream = self._add_stream(width, height, rate)
            self._time_base = fractions.Fraction(1) / rate
        except Exception:
            # Leaving a half-open container behind would write a header-only
            # file at close and hide the real error
            self._container.close()
            self._container = None
            self._stream = None
            raise

    def close(self):
        """Flush the encoder, write the trailer, and let the file go."""
        if self._container is None:
            return

        try:
            if self._stream is not None:
                for packet in self._stream.encode():
                    self._container.mux(packet)
        except Exception:
            logger.exception("could not flush the encoder for %s",
                             self._filename)

        try:
            self._container.close()
        finally:
            self._container = None
            self._stream = None
            self._graph = None
            self._graph_source = None
            self._graph_sink = None
            self._source_format = None

    def good(self):
        return self._container is not None

    # ------------------------------------------------------------------
    # Writing

    def add_image(self, image, ts=None):
        if self._container is None:
            raise RuntimeError(
                "add_image() called before a successful open()")

        array = image.image().asarray()

        # The timestamp is deliberately unused: the arrow numbered its frames
        # by how many it had been given, and a stream whose rate is fixed at
        # open cannot express anything else
        frame = self._convert(array)

        # In the stream's own units, so that presentation times come out at
        # the configured rate. Without a time base on the frame the encoder
        # reads the pts as seconds and the video plays back at one frame a
        # second
        frame.time_base = self._time_base
        frame.pts = self._count

        for packet in self._stream.encode(frame):
            self._container.mux(packet)

        self._count += 1

    def add_metadata(self, md):
        # The arrow left this unimplemented; VIAME writes no metadata streams
        pass

    # ------------------------------------------------------------------

    def _resolve(self, settings):
        """Geometry and frame rate, from the settings then the config."""
        width = height = 0
        rate = None

        if settings is not None:
            width = int(settings.width())
            height = int(settings.height())

            if settings.frame_rate() > 0:
                rate = fractions.Fraction(
                    settings.frame_rate()).limit_denominator(1001)

        width = width or self._width
        height = height or self._height

        if rate is None and self._frame_rate_num > 0:
            rate = fractions.Fraction(self._frame_rate_num,
                                      self._frame_rate_den or 1)

        if width <= 0 or height <= 0 or rate is None:
            raise RuntimeError(
                "video output requires width, height, and frame rate to be "
                "specified prior to calling open()")

        return width, height, rate

    def _add_stream(self, width, height, rate):
        stream = None
        tried = []

        for name in self._codec_choices():
            tried.append(name)
            try:
                stream = self._container.add_stream(name, rate=rate)
            except Exception as exc:
                logger.debug("codec %s unusable here: %s", name, exc)
                continue
            break

        if stream is None:
            raise RuntimeError(
                "could not open video with any known output codec; {} were "
                "tried: {}".format(len(tried), ", ".join(tried)))

        stream.width = width
        stream.height = height

        pixel_format = self._supported_pixel_format(stream)
        if pixel_format:
            stream.pix_fmt = pixel_format

        # Unconditionally, including the zero that means "unset": PyAV gives
        # a new stream a default bit rate, which puts x264 and x265 into
        # average bit rate mode, where the arrow left the field at zero and
        # got their default constant quality instead
        stream.bit_rate = int(self._bitrate)

        if self._crf:
            stream.options = {"crf": str(self._crf)}

        return stream

    def _supported_pixel_format(self, stream):
        """The configured pixel format, if this encoder has it.

        The arrow ran the request through `avcodec_find_best_pix_fmt_of_list`
        and let the encoder choose when the name was unusable, rather than
        failing at the first frame. An empty setting means the same thing it
        did there: leave the encoder to its own preference.
        """
        if not self._pixel_format:
            return None

        formats = getattr(stream.codec, "video_formats", None) or ()
        names = [entry.name for entry in formats]

        if names and self._pixel_format not in names:
            logger.warning(
                "%s cannot encode %s; letting it choose its own format",
                stream.codec.name, self._pixel_format)
            return None

        return self._pixel_format

    def _codec_choices(self):
        """Codecs to try, in the arrow's priority order.

        The arrow asked libavformat which codecs the container would take and
        sorted the answer; PyAV exposes no such query, so the preferred two
        are tried in order and a container that takes neither has to be told
        what to use through `codec_name`.
        """
        if self._codec_name:
            return (self._codec_name,) + PREFERRED_CODECS

        return PREFERRED_CODECS

    def _convert(self, array):
        """The frame the encoder wants, out of the image vital handed us."""
        import av

        source, prepared = _source_frame(array)
        shape = (source, prepared.shape[1], prepared.shape[0])

        if shape != self._source_format:
            self._build_graph(*shape)

        frame = av.VideoFrame.from_ndarray(prepared, format=source)
        frame.pts = None

        self._graph_source.push(frame)

        return self._graph_sink.pull()

    def _build_graph(self, source, width, height):
        """buffer -> scale -> format, so swscale runs with the arrow's flags.

        `format` after `scale` is not a second conversion: libavfilter
        negotiates the output format backwards, so scale converts straight to
        it under the flags given here.
        """
        import av

        graph = av.filter.Graph()

        buffer = graph.add(
            "buffer",
            "width={}:height={}:pix_fmt={}:time_base=1/1".format(
                width, height, source))

        out_range = ("full" if self._stream.pix_fmt in FULL_RANGE_FORMATS
                     else "limited")
        scale = graph.add(
            "scale",
            "{}:in_range=full:out_range={}".format(SCALE_FLAGS, out_range))

        fmt = graph.add("format", self._stream.pix_fmt)
        sink = graph.add("buffersink")

        buffer.link_to(scale)
        scale.link_to(fmt)
        fmt.link_to(sink)
        graph.configure()

        self._graph = graph
        self._graph_source = buffer
        self._graph_sink = sink
        self._source_format = (source, width, height)


def _source_frame(array):
    """The pixel format and array to hand libavfilter.

    Vital images are 1, 3 or 4 channel and 8 or 16 bit; the formats below are
    the ones `vital_to_frame_pix_fmt` maps those to for a packed image, which
    is what an image arriving from a process is.
    """
    array = np.asarray(array)

    if array.ndim == 2:
        array = array[:, :, np.newaxis]

    if array.dtype == np.uint8:
        formats = {1: "gray", 3: "rgb24", 4: "rgba"}
    elif array.dtype == np.uint16:
        formats = {1: "gray16le", 3: "rgb48le", 4: "rgba64le"}
    else:
        raise RuntimeError(
            "cannot write an image of type {}".format(array.dtype))

    depth = array.shape[2]

    if depth not in formats:
        raise RuntimeError(
            "cannot write an image with {} channels".format(depth))

    return formats[depth], np.ascontiguousarray(array.squeeze(2)
                                                if depth == 1 else array)


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


# The name arrows/ffmpeg registered its writer under, which every shipped
# transcode pipeline asks for; see the note on the reader's aliases.
class FFmpegVideoOutput(PyAVVideoOutput):
    """The name `arrows/ffmpeg` registered its writer under."""


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        PyAVVideoOutput, "pyav", "Write a video with PyAV")
    register_vital_algorithm(
        FFmpegVideoOutput, "ffmpeg", "Write a video with PyAV")
