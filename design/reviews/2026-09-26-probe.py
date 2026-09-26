"""Read-only conversion review probes; run after sourcing setup_viame.sh.

Python implementations are loaded from the reviewed commit. Native kernels
come from the installed build: this script never builds or installs anything.
Temporary media are removed on exit. Results are diagnostic, not pass/fail
tests; a repaired native build should produce different results.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType

import av
import cv2
import numpy as np
from PIL import Image as PILImage
from viame import image_kernels as kernels

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "2ca6de3f8"


def committed(path):
    return subprocess.check_output(
        ["git", "show", COMMIT + ":" + path], cwd=ROOT)


def module(name, path):
    result = ModuleType(name)
    result.__file__ = str(ROOT / path)
    exec(compile(committed(path), result.__file__, "exec"), result.__dict__)
    return result


def median_ms(function):
    function()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return round(float(np.median(samples)), 3)


def heartbeat(function):
    ticks = []
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.002)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.015)
    try:
        start = time.perf_counter()
        function()
        end = time.perf_counter()
    finally:
        stop.set()
        thread.join()
    return dict(duration_ms=round((end - start) * 1000, 3),
                ticks_during_call=sum(start < tick < end for tick in ticks),
                largest_gap_ms=round(float(max(np.diff(ticks + [end]))) * 1000, 3))


def main():
    cv2.setNumThreads(1)
    out = dict(commit=COMMIT, native_package=kernels.__file__,
               opencv=cv2.__version__, opencv_threads=cv2.getNumThreads())
    ops = module("review_imageops", "library/utilities/imageops.py")
    video = module("review_video", "library/video_io/pyav_video_input.py")
    predictor = module("review_onnx", "library/object_detectors/onnx/onnx_predictor.py")

    checker = np.array([[0, 255], [255, 0]], np.uint8)
    out["bilinear"] = dict(lite=kernels.resize(checker, 4, 4).tolist(),
                           reference=cv2.resize(checker, (4, 4)).tolist())
    impulse = np.array([[0, 0, 255, 0, 0]], np.uint8)
    out["area"] = dict(lite=kernels.resize_area(impulse, 3, 1).tolist(),
                       reference=cv2.resize(impulse, (3, 1), interpolation=cv2.INTER_AREA).tolist(),
                       float_constant=kernels.resize_area(np.ones((4, 4), np.float32), 2, 2).tolist())
    # Exercise preprocessing without loading an ONNX model or inference engine.
    pred = predictor.OnnxPredictor.__new__(predictor.OnnxPredictor)
    pred._eval_w = pred._eval_h = 4
    pred._interp_name = "nearest"
    pred._scale, pred._mean, pred._std = 1.0, 0.0, 1.0
    out["onnx_nearest"] = pred._preprocess(checker)[0, 0].tolist()

    with tempfile.TemporaryDirectory(prefix="viame-conversion-review-") as scratch:
        scratch = Path(scratch)
        rgb_path = scratch / "rgb.png"
        PILImage.fromarray(np.full((32, 32, 3), 100, np.uint8)).save(rgb_path)
        image = np.ascontiguousarray(ops.read_unchanged(rgb_path))
        out["read_writeable"] = image.flags.writeable
        try:
            kernels.fill_polygon(image, [(0., 0.), (5., 0.), (5., 5.)], 0)
            out["draw_after_read"] = "success"
        except Exception as exc:
            out["draw_after_read"] = str(exc)
        for name, array in (
            ("uint16", np.array([[0, 256, 2000, 65535]], np.uint16)),
            ("rgba", np.array([[[10, 20, 30, 40], [50, 60, 70, 80]]], np.uint8)),
        ):
            path = scratch / (name + ".png")
            returned = ops.write_image(path, array)
            restored = ops.read_unchanged(path)
            out[name + "_write"] = dict(returned=returned, dtype=str(restored.dtype),
                                        pixels=restored.tolist())

        # The real trainer adapter, with real bound detection/keypoint types.
        from viame.types import BoundingBoxD, DetectedObject, DetectedObjectSet, Point2d
        import viame.utilities
        viame.utilities.imageops = ops
        sys.modules["viame.utilities.imageops"] = ops
        common = module("viame.classifiers.sleap.sleap_common",
                        "library/classifiers/sleap/sleap_common.py")
        sys.modules[common.__name__] = common
        trainer_mod = module("review_sleap", "library/classifiers/sleap/sleap_trainer.py")
        trainer = trainer_mod.SleapTrainer()
        trainer.options.update(train_directory=str(scratch / "sleap"),
                               crop_width=32, crop_height=32, crop_padding=1.0)
        detection = DetectedObject(BoundingBoxD(0, 0, 32, 32))
        detection.add_keypoint("head", Point2d(np.array([10., 10.])))
        detections = DetectedObjectSet()
        detections.add(detection)
        try:
            trainer.add_data_from_disk(None, [str(rgb_path)], [detections], [], [])
            out["sleap_export"] = "success"
        except Exception as exc:
            out["sleap_export"] = dict(error=type(exc).__name__ + ": " + str(exc),
                                       crops_written=len(list((scratch / "sleap").rglob("*.png"))))

        clip = scratch / "clip.mp4"
        clip.write_bytes(committed("tests/golden/inputs/clip.mp4"))
        reader = video.PyAVVideoInput()
        reader.open(str(clip))
        try:
            reader.next_frame()
            reader.frame_timestamp()
            reader.seek_frame(15)
            seek = dict(format=reader._frame.format.name,
                        time=reader.frame_timestamp().get_time_seconds())
            try:
                seek["shape"] = reader.frame_image().asarray().shape
            except Exception as exc:
                seek["image_error"] = str(exc)
            reader.next_frame()
            seek["next_time"] = reader.frame_timestamp().get_time_seconds()
            out["seek"] = seek
        finally:
            reader.close()

        clip = scratch / "count.mkv"
        with av.open(str(clip), "w") as writer:
            stream = writer.add_stream("ffv1", rate=10)
            stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv444p"
            for index in range(12):
                frame = av.VideoFrame.from_ndarray(
                    np.full((24, 32, 3), index * 20, np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    writer.mux(packet)
            for packet in stream.encode():
                writer.mux(packet)
        reader = video.PyAVVideoInput()
        reader._filter_desc = ""
        reader.open(str(clip))
        try:
            for _ in range(3):
                reader.next_frame()
                reader.frame_timestamp()
            count = dict(actual=12, reported_after_three=reader.num_frames())
            reader.next_frame()
            count["next_time"] = reader.frame_timestamp().get_time_seconds()
            out["num_frames"] = count
        finally:
            reader.close()

    rgb = np.random.default_rng(20260926).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    gray = rgb[:, :, 0].copy()
    cases = (
        ("resize_1080_to_640", lambda: kernels.resize(rgb, 640, 360),
         lambda: cv2.resize(rgb, (640, 360))),
        ("area_1080_to_640", lambda: kernels.resize_area(rgb, 640, 360),
         lambda: cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)),
        ("gray_1080", lambda: kernels.to_gray(rgb), lambda: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)),
        ("gaussian_17_gray", lambda: kernels.gaussian_blur(gray, 17, 2.5),
         lambda: cv2.GaussianBlur(gray, (17, 17), 2.5)),
    )
    out["timings_ms"] = {name: dict(lite=median_ms(lite), opencv=median_ms(reference))
                         for name, lite, reference in cases}
    out["heartbeat"] = dict(lite=heartbeat(lambda: kernels.gaussian_blur(gray, 31, 5.)),
                            opencv=heartbeat(lambda: cv2.GaussianBlur(gray, (31, 31), 5.)))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
