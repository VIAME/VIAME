"""Drive a video_input over the golden clip and describe what came out.

What matters for a reader is which frames it yields, in what order, with what
timestamps: comparing whole decoded frames would mostly be measuring the codec
rather than the reader, so the images are recorded as digests.
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


CLIP = os.path.join(HERE, "inputs", "clip.mp4")


def run_isolated(impl, config, path, action="read"):
    """Do the work in a fresh process and bring the answer back as JSON.

    The C++ FFmpeg reader deadlocks inside `open` if PyAV has already used
    libav in the same process: both link the same libraries, and something in
    their shared global state does not survive the other having initialised
    it. That only matters while both readers are registered, which is the
    transition this phase is in the middle of, but it means a test comparing
    the two cannot run them side by side. Running each in its own process is
    also closer to how a pipeline uses a reader.
    """
    import pipeline_runner

    script = (
        "import json, sys; sys.path.insert(0, {here!r});"
        "import video_runner;"
        "video_runner.load_modules();"
        "print(json.dumps(video_runner.{action}({impl!r}, {config!r},"
        " {path!r})))"
    ).format(here=HERE, action=action, impl=impl, config=config, path=path)

    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=dict(pipeline_runner.sourced_environment(),
                 KWIVER_DEFAULT_LOG_LEVEL="error"))

    if result.returncode != 0:
        raise AssertionError("{} in a child process exited {}:\n{}".format(
            impl, result.returncode,
            result.stderr.decode("utf-8", "replace")[-2000:]))

    return json.loads(result.stdout.decode("utf-8").strip().splitlines()[-1])


def load_modules():
    import kwiver.vital.modules as modules

    modules.load_known_modules()


def is_registered(impl):
    from kwiver.vital.algo import VideoInput

    return impl in VideoInput.registered_names()


def decode_only(impl, config, path):
    """Count frames as fast as the reader will give them.

    No numpy conversion and no hashing: those cost more than the decode on a
    small frame and would make a throughput figure meaningless.
    """
    from kwiver.vital.algo import VideoInput

    algorithm = VideoInput.create(impl)

    if config:
        block = algorithm.get_configuration()
        for key, value in sorted(config.items()):
            block.set_value(key, str(value))
        algorithm.set_configuration(block)

    algorithm.open(path)

    count = 0

    try:
        while algorithm.next_frame():
            algorithm.frame_image()
            count += 1
    finally:
        algorithm.close()

    return count


def throughput(impl, config, path):
    """Frames per second decoding \p path, best of three passes.

    Timed inside this process: spawning one is several seconds, which would
    swamp the measurement on a short clip.
    """
    import time

    best = 0.0

    for _ in range(3):
        start = time.perf_counter()
        count = decode_only(impl, config, path)
        elapsed = time.perf_counter() - start

        if elapsed > 0:
            best = max(best, count / elapsed)

    return round(best, 1)


def seek_probe(impl, config, path):
    """Seek to a few frames and report where each landed.

    Returns [(target, landed_on, next_frame)], or None if the
    implementation is not registered in this build.
    """
    from kwiver.vital.algo import VideoInput

    if not is_registered(impl):
        return None

    reader = VideoInput.create(impl)
    reader.open(path)

    try:
        total = 0
        while reader.next_frame():
            total += 1

        reader.close()
        reader = VideoInput.create(impl)
        reader.open(path)

        landings = []

        for target in (1, 7, total // 2, total - 1):
            if not reader.seek_frame(target):
                raise AssertionError(
                    "{} could not seek to frame {}".format(impl, target))

            landed = reader.frame_timestamp().get_frame()
            following = (reader.frame_timestamp().get_frame()
                         if reader.next_frame() else None)
            landings.append((target, landed, following))

        return landings
    finally:
        reader.close()


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
                "dtype": str(array.dtype) if array is not None else None,
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(array).tobytes()
                ).hexdigest()[:16] if array is not None else None,
            })
    finally:
        algorithm.close()

    return frames
