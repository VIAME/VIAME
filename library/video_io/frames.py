# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Video frames as RGB arrays, for a script rather than a pipeline.

What a tool used `cv2.VideoCapture` for. `pyav_video_input` beside this is the
algorithm a pipeline selects, and it is the wrong shape for a script: it wants
the plugin framework loaded, it returns `ImageContainer`s, and it carries the
seeking, timestamp and metadata machinery a `video_input` has to. This is the
read loop on its own.

**The pixels are the same ones a pipeline sees.** The scale flags are
`pyav_video_input`'s, which mirror `arrows/ffmpeg/ffmpeg_convert_image.cxx`:
without them swscale interpolates chroma rather than taking the nearest
sample and writes limited-range RGB, which differs by up to three counts on
nearly every pixel. Frames come back **RGB**, not `cv2.VideoCapture`'s BGR,
because everything on this branch is RGB internally.

    for frame, number in read_frames("dive.mp4"):
        ...

`number` is one-based and counts every decoded frame, not just the ones
yielded, so it lines up with what a viewer calls that frame.
"""

import numpy as np

# `pyav_video_input`'s, so a calibration reading a video sees the pixels a
# pipeline reading the same video would.
SCALE_FLAGS = ("flags=full_chroma_int+full_chroma_inp+accurate_rnd+bitexact"
               "+neighbor:out_range=full")


def _decoder(path):
    """The container and its first video stream."""
    import av

    container = av.open(path)
    streams = [s for s in container.streams if s.type == "video"]

    if not streams:
        container.close()
        raise ValueError("no video stream in '{}'".format(path))

    stream = streams[0]
    stream.thread_type = "AUTO"
    return container, stream


def count_frames(path):
    """What the container claims, or 0 when it claims nothing.

    `cv2.CAP_PROP_FRAME_COUNT` with the same caveat it had: a container that
    does not record a frame count reports zero rather than being scanned, and
    a caller uses it for a progress bar rather than for correctness.
    """
    container, stream = _decoder(path)
    try:
        return int(stream.frames or 0)
    finally:
        container.close()


def read_frames(path):
    """Yield `(rgb_array, frame_number)` for every frame of a video.

    The array is (h, w, 3) uint8, C contiguous, and freshly owned -- a caller
    may keep or modify it.
    """
    import av

    container, stream = _decoder(path)

    try:
        graph = av.filter.Graph()
        source = graph.add_buffer(template=stream)
        scale = graph.add("scale", SCALE_FLAGS)
        # Packed rather than planar, because a numpy consumer wants (h, w, 3);
        # the conversion after the scale is exact either way.
        packed = graph.add("format", "rgb24")
        sink = graph.add("buffersink")

        source.link_to(scale)
        scale.link_to(packed)
        packed.link_to(sink)
        graph.configure()

        number = 0

        def drain():
            nonlocal number
            while True:
                try:
                    out = graph.pull()
                except (av.error.BlockingIOError, av.error.EOFError):
                    return
                number += 1
                yield np.ascontiguousarray(out.to_ndarray()), number

        for frame in container.decode(stream):
            graph.push(frame)
            for pair in drain():
                yield pair

        # A filter may hold a frame back; flushing asks for the rest.
        try:
            graph.push(None)
        except av.error.EOFError:
            pass
        for pair in drain():
            yield pair
    finally:
        container.close()


def video_size(path):
    """`(width, height)` of the first video stream, without decoding a frame.

    `cv2.CAP_PROP_FRAME_WIDTH` and `_HEIGHT`.
    """
    container, stream = _decoder(path)
    try:
        return int(stream.codec_context.width), int(
            stream.codec_context.height)
    finally:
        container.close()


def frame_rate(path):
    """Frames per second, or 0.0 when the container does not say.

    `cv2.CAP_PROP_FPS`. The average rate, which is what a variable frame rate
    container can honestly answer and what OpenCV reports for one.
    """
    container, stream = _decoder(path)
    try:
        rate = stream.average_rate or stream.guessed_rate
        return float(rate) if rate else 0.0
    finally:
        container.close()


class FrameReader:
    """Random access to a video's frames, which is what `cv2.VideoCapture` is.

    `read_frames` above is the loop, and a loop is all most callers want. This
    is for the ones that seek -- `mmcv.VideoReader` indexes and slices -- and
    it is deliberately the same decode: the frames are `read_frames`' frames,
    RGB and bit identical to the ones a pipeline sees.

    **Seeking backwards restarts the decode.** PyAV can seek to a keyframe
    cheaply, but landing on the requested frame from there means decoding
    forward anyway and the keyframe interval is the container's business, not
    ours. Restarting is exact, it is what a caller stepping backwards through
    a video is really asking for, and it is O(position). Seeking forwards
    decodes and discards, which is what OpenCV does too.
    """

    def __init__(self, path):
        self._path = str(path)
        self._frames = None
        self._position = 0
        self._width, self._height = video_size(self._path)
        self._rate = frame_rate(self._path)
        self._count = count_frames(self._path)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def frame_rate(self):
        return self._rate

    @property
    def frame_count(self):
        """What the container claims; 0 when it claims nothing."""
        return self._count

    @property
    def position(self):
        """The number of the frame `read` will return next, zero based."""
        return self._position

    def read(self):
        """The next frame as an (h, w, 3) uint8 RGB array, or None at the end."""
        if self._frames is None:
            self._frames = read_frames(self._path)

        for array, _number in self._frames:
            self._position += 1
            return array

        return None

    def seek(self, number):
        """Position the reader so that `read` returns frame `number`."""
        number = max(int(number), 0)

        if number < self._position or self._frames is None:
            self.close()
            self._frames = read_frames(self._path)
            self._position = 0

        while self._position < number:
            if self.read() is None:
                return

    def close(self):
        if self._frames is not None:
            self._frames.close()
            self._frames = None

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()


class FrameWriter:
    """Encode RGB frames to a video file, which is `cv2.VideoWriter`.

    The encoder settings are `pyav_video_output`'s -- its codec preference
    order, its scale flags, and its habit of setting the bit rate explicitly
    so that x264 stays in constant quality mode rather than picking up PyAV's
    default average bit rate. What this is not is that algorithm: no plugin
    framework, no `ImageContainer`, and the frame rate is fixed when the file
    is opened rather than configured.
    """

    #: In preference order, as `pyav_video_output._codec_choices` has it.
    CODECS = ("libx264", "libopenh264", "h264", "mpeg4")

    def __init__(self, path, width, height, rate, codec=None, crf=23):
        import av

        self._path = str(path)
        self._width = int(width)
        self._height = int(height)
        self._container = av.open(self._path, mode="w")
        self._stream = None
        self._graph = None
        self._source_format = None

        tried = []

        for name in ((codec,) if codec else self.CODECS):
            tried.append(name)
            try:
                self._stream = self._container.add_stream(
                    name, rate=_exact_rate(rate))
            except Exception:
                continue
            break

        if self._stream is None:
            self._container.close()
            raise RuntimeError(
                "no usable output codec; tried {}".format(", ".join(tried)))

        self._stream.width = self._width
        self._stream.height = self._height
        self._stream.pix_fmt = "yuv420p"
        self._stream.bit_rate = 0

        if crf:
            self._stream.options = {"crf": str(int(crf))}

    def write(self, rgb):
        """Encode one (h, w, 3) uint8 RGB frame."""
        import av

        array = np.ascontiguousarray(rgb)

        if array.ndim == 2:
            array = np.repeat(array[:, :, np.newaxis], 3, axis=2)

        if array.dtype != np.uint8 or array.shape[2] != 3:
            raise ValueError(
                "FrameWriter takes 8-bit RGB, got {} with shape {}".format(
                    array.dtype, array.shape))

        height, width = array.shape[:2]
        self._ensure_graph(width, height)
        self._graph_source.push(
            av.VideoFrame.from_ndarray(array, format="rgb24"))

        while True:
            try:
                converted = self._graph_sink.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                break
            for packet in self._stream.encode(converted):
                self._container.mux(packet)

    def _ensure_graph(self, width, height):
        """buffer -> scale -> format, with `pyav_video_output`'s flags."""
        import av

        if self._source_format == (width, height):
            return

        graph = av.filter.Graph()
        source = graph.add(
            "buffer",
            "width={}:height={}:pix_fmt=rgb24:time_base=1/1".format(
                width, height))
        # `out_range` is limited because the output format is yuv420p, which
        # is a limited-range format; `pyav_video_output` picks the same way.
        scale = graph.add(
            "scale", "{}:in_range=full:out_range=limited".format(
                SCALE_FLAGS.replace("out_range=full", "").rstrip(":")))
        fmt = graph.add("format", self._stream.pix_fmt)
        sink = graph.add("buffersink")

        source.link_to(scale)
        scale.link_to(fmt)
        fmt.link_to(sink)
        graph.configure()

        self._graph = graph
        self._graph_source = source
        self._graph_sink = sink
        self._source_format = (width, height)

    def close(self):
        if self._container is None:
            return

        try:
            if self._stream is not None:
                for packet in self._stream.encode():
                    self._container.mux(packet)
        finally:
            self._container.close()
            self._container = None

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()


def _exact_rate(rate):
    """A frame rate PyAV will take, as a fraction when it is not an integer."""
    from fractions import Fraction

    value = float(rate)

    if value <= 0.0:
        raise ValueError("a frame rate must be positive, got {}".format(rate))

    if value == int(value):
        return int(value)

    return Fraction(value).limit_denominator(1001)
