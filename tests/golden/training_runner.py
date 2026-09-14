"""Drive a windowed `train_detector` over the golden scene.

Shared by the recorder and the golden test. The chips themselves are not
committed -- a few dozen 400 by 300 colour chips per variant would be most of
the golden tree -- but a digest of each chip's pixels and shape is, and the
manifests are recorded in full, so a chip that moved, changed or went missing
is still a failure that names the chip.
"""

import glob
import hashlib
import os

import numpy as np

import imageio_utils
import training_cases


def _truth(name):
    from kwiver.vital.types import (BoundingBoxD, DetectedObject,
                                    DetectedObjectSet, DetectedObjectType)

    width, height = training_cases.SIZES[name]
    out = DetectedObjectSet()

    for min_x, min_y, max_x, max_y, category in training_cases.TRUTH:
        if max_x > width or max_y > height:
            continue

        out.add(DetectedObject(BoundingBoxD(min_x, min_y, max_x, max_y), 1.0,
                               DetectedObjectType(category, 1.0)))

    return out


def _digest(array):
    array = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode())
    hasher.update(str(array.dtype).encode())
    hasher.update(array.tobytes())
    return np.frombuffer(hasher.digest(), dtype=np.uint8).copy()


def _parse_manifests(train_dir, chip_dir):
    """The manifests as (files, boxes) arrays, frames in tag order.

    files: one row per image handed on -- frame index, whether it is a chip
    (1) or the original left where it was (0), chip index within its frame
    (-1 for an original), detection count.
    boxes: one row per detection -- row index into files, category id (-1 if
    the category is not a synthetic id), min_x, min_y, max_x, max_y, score.
    """
    files = []
    boxes = []

    for path in sorted(glob.glob(os.path.join(train_dir, "**", "*.manifest"),
                                 recursive=True)):
        frame = int(os.path.basename(path).split("_", 1)[0])

        with open(path) as handle:
            lines = handle.read().splitlines()

        assert lines and lines[0].startswith("VIAME_CHIP_MANIFEST"), path

        for line in lines[1:]:
            fields = line.split()

            if fields[0] == "F":
                name = fields[1]
                is_chip = os.path.dirname(os.path.abspath(name)) == chip_dir
                index = (int(os.path.splitext(name)[0].rsplit("_", 1)[1])
                         if is_chip else -1)
                files.append([frame, int(is_chip), index, int(fields[2])])

            elif fields[0] == "D":
                category = int(fields[1]) if fields[1].lstrip("-").isdigit() \
                    else -1
                boxes.append([len(files) - 1, category] +
                             [float(value) for value in fields[2:7]])

    return (np.array(files, dtype=np.int64).reshape(len(files), 4),
            np.array(boxes, dtype=np.float64).reshape(len(boxes), 7))


def run(impl, config, image_paths, work_dir):
    """Chip the scene with `impl`, returning the result as named arrays."""
    from kwiver.vital.algo import TrainDetector

    import runner

    algorithm = TrainDetector.create(impl)

    if algorithm is None:
        raise RuntimeError("train_detector '{}' is not registered".format(impl))

    train_dir = os.path.join(work_dir, "train")
    config = dict(config)
    config["train_directory"] = train_dir
    config["trainer:frame_diff:output_directory"] = os.path.join(
        work_dir, "models")

    # frame_diff makes its directories relative to the working directory
    previous = os.getcwd()
    os.chdir(work_dir)

    try:
        runner._configure(algorithm, config)
        algorithm.add_data_from_disk(
            None, list(image_paths),
            [_truth(os.path.splitext(os.path.basename(path))[0])
             for path in image_paths],
            [], [])
    finally:
        os.chdir(previous)

    chip_dir = os.path.abspath(os.path.join(train_dir, "cached_chips"))
    files, boxes = _parse_manifests(train_dir, chip_dir)

    out = {"files": files, "boxes": boxes}

    for path in sorted(glob.glob(os.path.join(chip_dir, "*.png"))):
        tag = os.path.splitext(os.path.basename(path))[0]
        out["chip_" + tag] = _digest(imageio_utils.load(path))

    return out
