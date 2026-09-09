#!/usr/bin/env python3
"""Record what the video reader does, before it is replaced.

Run against an install that still has the reader being replaced:

    source <install>/setup_viame.sh
    python3 tests/golden/record_video.py

Frames are recorded as digests rather than images: what matters for a reader
is which frames it yields, in what order, and with what timestamps, and
committing decoded 1080p frames would mostly be measuring the codec.

The throughput figure is what phase 4 has to stay within reach of. It is
measured on a clip generated here rather than committed, because a 1080p clip
long enough to time is far too large for the repository; it is built from the
committed fixtures so the content is at least fixed.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pipeline_runner              # noqa: E402
import video_cases                  # noqa: E402
import video_runner                 # noqa: E402

INPUTS = os.path.join(HERE, "inputs")
GROUP = os.path.join(HERE, "video")

# The clips the manifest covers, by the property each one is there to pin.
CLIPS = {
    "clip.mp4": "8 bit h264, constant frame rate",
    "clip_10bit.mp4": "10 bit h264, so the reader's bit depth handling shows",
    "clip_vfr.mp4": "variable frame rate, so timestamps cannot be inferred "
                    "from the frame number",
}

THROUGHPUT_FRAMES = 120
THROUGHPUT_SIZE = "1920x1080"


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def describe(impl, clip):
    """Frame count, timestamps, and digests of the first, middle and last."""
    frames = video_runner.read(impl, {}, os.path.join(INPUTS, clip))

    if not frames:
        raise RuntimeError("{} yielded no frames from {}".format(impl, clip))

    count = len(frames)
    sampled = sorted({0, count // 2, count - 1})

    return {
        "frames": count,
        "timestamps": [frame["time"] for frame in frames],
        "frame_numbers": [frame["frame"] for frame in frames],
        "shape": frames[0]["shape"],
        "dtype": frames[0]["dtype"],
        "digests": {
            str(index + 1): frames[index]["sha256"] for index in sampled
        },
    }


def build_throughput_clip(path):
    """A 1080p clip built from the committed fixtures, not committed itself."""
    listing = os.path.join(os.path.dirname(path), "throughput_list.txt")
    frames = sorted(
        os.path.join(INPUTS, name) for name in os.listdir(INPUTS)
        if name.startswith("frame_") and name.endswith(".png"))

    with open(listing, "w") as handle:
        for _ in range(THROUGHPUT_FRAMES // len(frames) + 1):
            for frame in frames:
                handle.write("file '{}'\n".format(frame))

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-r", "30", "-f", "concat", "-safe", "0", "-i", listing,
         "-vf", "scale=" + THROUGHPUT_SIZE.replace("x", ":"),
         "-frames:v", str(THROUGHPUT_FRAMES),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        check=True)


def measure_throughput(impl, path):
    """Decoded frames per second, best of three passes."""
    best = 0.0

    for _ in range(3):
        start = time.perf_counter()
        frames = video_runner.decode_only(impl, {}, path)
        elapsed = time.perf_counter() - start

        if elapsed > 0:
            best = max(best, frames / elapsed)

    return round(best, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", default="ffmpeg",
                        help="video_input and video_output implementation "
                             "to record")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing recording")
    args = parser.parse_args()

    manifest_path = os.path.join(GROUP, "manifest.json")

    if os.path.exists(manifest_path) and not args.force:
        print("already recorded; pass --force to re-record")
        return 1

    video_runner.load_modules()
    os.makedirs(GROUP, exist_ok=True)

    manifest = {
        "impl": args.impl,
        "recorded": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clips": {},
    }

    for clip, why in sorted(CLIPS.items()):
        manifest["clips"][clip] = dict(describe(args.impl, clip), why=why)
        print("{:16s} {} frames".format(clip, manifest["clips"][clip]["frames"]))

    import tempfile

    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "throughput.mp4")
        build_throughput_clip(path)

        manifest["throughput"] = {
            "frames_per_second": measure_throughput(args.impl, path),
            "resolution": THROUGHPUT_SIZE,
            "frames": THROUGHPUT_FRAMES,
            "codec": "h264",
            "note": "best of three passes, decode and frame_image only with "
                    "no pixel conversion, on a clip built from the "
                    "committed fixtures rather than committed itself",
        }

    print("throughput: {} fps at {}".format(
        manifest["throughput"]["frames_per_second"], THROUGHPUT_SIZE))

    manifest["pipelines"] = {}

    for pipeline in video_cases.VIDEO_PIPELINES:
        written = pipeline_runner.run_video(pipeline)
        manifest["pipelines"][pipeline] = written
        print("{:24s} {} frames, {} s".format(
            pipeline, written["frames"], written["duration"]))

    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
