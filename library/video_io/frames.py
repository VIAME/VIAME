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
                except Exception:
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
        except Exception:
            pass
        for pair in drain():
            yield pair
    finally:
        container.close()
