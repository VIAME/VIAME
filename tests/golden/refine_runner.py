"""Drive a `refine_detections` implementation over the golden fixtures.

Shared by the recorder and the golden test. What is recorded is everything a
refiner can change about a detection -- the box, the confidence, the type
scores, the notes, the keypoints and the **mask**, which for the two
segmenters is the whole output.
"""

import numpy as np

import refine_cases
import runner


def build_detections():
    """The input set, from `refine_cases.DETECTIONS`."""
    from kwiver.vital.types import (BoundingBoxD, DetectedObject,
                                    DetectedObjectSet, DetectedObjectType,
                                    Image, ImageContainer)

    out = DetectedObjectSet()

    for entry in refine_cases.DETECTIONS:
        box = BoundingBoxD(*[float(value) for value in entry["bbox"]])

        detection = DetectedObject(
            box, float(entry["confidence"]),
            DetectedObjectType(entry["type"], float(entry["confidence"])))

        if entry["mask"] == "ellipse":
            mask = _ellipse_mask(entry["bbox"])
            detection.mask = ImageContainer(Image(mask))

        out.add(detection)

    return out


def _ellipse_mask(bbox):
    """An ellipse inscribed in the box, as the byte mask vital carries.

    Box-sized, as a detection's mask is: the mask's origin is the box's
    upper left, not the image's.
    """
    width = int(round(bbox[2] - bbox[0]))
    height = int(round(bbox[3] - bbox[1]))

    ys, xs = np.mgrid[0:height, 0:width]

    centre_x = (width - 1) / 2.0
    centre_y = (height - 1) / 2.0
    radius_x = max(1.0, centre_x * refine_cases.ELLIPSE_FILL)
    radius_y = max(1.0, centre_y * refine_cases.ELLIPSE_FILL)

    inside = (((xs - centre_x) / radius_x) ** 2 +
              ((ys - centre_y) / radius_y) ** 2) <= 1.0

    return np.ascontiguousarray(inside.astype(np.uint8))


def describe(detections):
    """A refined detection set as named arrays.

    Arrays rather than JSON because the masks are the interesting part and a
    mask is an image. One `mask_<n>` per detection, since they are different
    sizes; `mask_present` says which detections have one at all, so a refiner
    that stopped producing masks is a failure rather than a quietly shorter
    dictionary.
    """
    boxes = []
    confidences = []
    present = []
    keypoints = []
    out = {}

    for index, detection in enumerate(detections or []):
        box = detection.bounding_box
        boxes.append([box.min_x(), box.min_y(), box.max_x(), box.max_y()])
        confidences.append(detection.confidence)

        mask = detection.mask
        array = None if mask is None else mask.asarray()

        present.append(1 if array is not None else 0)

        if array is not None:
            if array.ndim == 3:
                array = array[:, :, 0]
            out["mask_{}".format(index)] = np.ascontiguousarray(
                array.astype(np.int64))

        for name in sorted(detection.keypoints or {}):
            point = detection.keypoints[name]
            location = np.asarray(point.value, dtype=np.float64)
            keypoints.append([index, _keypoint_id(name),
                              location[0], location[1]])

    out["boxes"] = np.array(boxes, dtype=np.float64).reshape(len(boxes), 4)
    out["confidences"] = np.array(confidences, dtype=np.float64)
    out["mask_present"] = np.array(present, dtype=np.int64)
    out["keypoints"] = np.array(keypoints, dtype=np.float64).reshape(
        len(keypoints), 4)

    return out


# Keypoint names, as the small integers an array can hold. Recorded rather
# than hashed so a name that changes is a failure and not a silent shuffle.
KEYPOINT_NAMES = ("head", "tail", "left", "right", "top", "bottom",
                  "center", "centroid", "p1", "p2")


def _keypoint_id(name):
    if name in KEYPOINT_NAMES:
        return KEYPOINT_NAMES.index(name)

    raise KeyError(
        "unrecorded keypoint name '{}'; add it to KEYPOINT_NAMES".format(name))


def run(impl, config, array):
    """Refine the fixture's detections, returning them as named arrays."""
    from kwiver.vital.algo import RefineDetections
    from kwiver.vital.types import Image, ImageContainer

    algorithm = RefineDetections.create(impl)

    if algorithm is None:
        raise RuntimeError(
            "refine_detections '{}' is not registered".format(impl))

    runner._configure(algorithm, config)

    refined = algorithm.refine(
        ImageContainer(Image(np.ascontiguousarray(array))),
        build_detections())

    return describe(refined)
