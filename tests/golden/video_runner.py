"""Drive a video_input over the golden clip and describe what came out.

What matters for a reader is which frames it yields, in what order, with what
timestamps: comparing whole decoded frames would mostly be measuring the codec
rather than the reader, so the images are recorded as digests.
"""

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CLIP = os.path.join(HERE, "inputs", "clip.mp4")


def load_modules():
    import kwiver.vital.modules as modules

    modules.load_known_modules()


def is_registered(impl):
    from kwiver.vital.algo import VideoInput

    return impl in VideoInput.registered_names()


def read(impl, config, path=None):
    """Open the clip and return one record per frame it yields."""
    from kwiver.vital.algo import VideoInput

    algorithm = VideoInput.create(impl)

    if algorithm is None:
        raise RuntimeError("video_input '{}' is not registered".format(impl))

    if config:
        block = algorithm.get_configuration()

        for key, value in sorted(config.items()):
            block.set_value(key, str(value))

        algorithm.set_configuration(block)

    algorithm.open(path or CLIP)

    frames = []

    try:
        while algorithm.next_frame():
            timestamp = algorithm.frame_timestamp()
            image = algorithm.frame_image()

            array = np.asarray(image.image().asarray()) if image else None

            frames.append({
                "frame": int(timestamp.get_frame())
                         if timestamp.has_valid_frame() else None,
                "time": round(float(timestamp.get_time_seconds()), 6)
                        if timestamp.has_valid_time() else None,
                "shape": list(array.shape) if array is not None else None,
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(array).tobytes()
                ).hexdigest()[:16] if array is not None else None,
            })
    finally:
        algorithm.close()

    return frames
