"""Drive one registered algorithm over the golden fixtures.

Shared by the recorder and the golden test so that both exercise the
implementation exactly the same way. Nothing here knows which implementation
is the old one and which is the replacement: that is the point.
"""

import os

import numpy as np


def load_modules():
    """Load every plugin, including the python-registered ones."""
    import kwiver.vital.modules as modules

    modules.load_known_modules()


def is_registered(kind, impl):
    """Whether this build has \p impl registered for that kind of case."""
    if kind == "image_filter":
        from kwiver.vital.algo import ImageFilter
        return impl in ImageFilter.registered_names()

    if kind in ("image_io", "decode", "round_trip"):
        from kwiver.vital.algo import ImageIO
        return impl in ImageIO.registered_names()

    if kind == "split_image":
        from kwiver.vital.algo import SplitImage
        return impl in SplitImage.registered_names()

    if kind == "detect_motion":
        from kwiver.vital.algo import DetectMotion
        return impl in DetectMotion.registered_names()

    if kind == "detect":
        from kwiver.vital.algo import ImageObjectDetector
        return impl in ImageObjectDetector.registered_names()

    return True


def _configure(algorithm, config):
    if not config:
        return

    block = algorithm.get_configuration()

    for key, value in sorted(config.items()):
        block.set_value(key, str(value))

    algorithm.set_configuration(block)


def make_image_filter(impl, config):
    from kwiver.vital.algo import ImageFilter

    algorithm = ImageFilter.create(impl)

    if algorithm is None:
        raise RuntimeError("image_filter '{}' is not registered".format(impl))

    _configure(algorithm, config)
    return algorithm


def run_image_filter(impl, config, arrays):
    """Filter `arrays` in order, returning one output array per input.

    Order matters: the temporal filters carry state from frame to frame.
    """
    from kwiver.vital.types import Image, ImageContainer

    algorithm = make_image_filter(impl, config)
    outputs = []

    for array in arrays:
        result = algorithm.filter(ImageContainer(Image(np.ascontiguousarray(array))))
        outputs.append(np.array(result.image().asarray(), copy=True))

    return outputs


def run_split_image(impl, config, arrays):
    """Split each array, returning the pieces of every input in order."""
    from kwiver.vital.algo import SplitImage
    from kwiver.vital.types import Image, ImageContainer

    algorithm = SplitImage.create(impl)

    if algorithm is None:
        raise RuntimeError("split_image '{}' is not registered".format(impl))

    _configure(algorithm, config)

    outputs = []
    for array in arrays:
        pieces = algorithm.split(
            ImageContainer(Image(np.ascontiguousarray(array))))
        outputs.append([np.array(piece.image().asarray(), copy=True)
                        for piece in pieces])

    return outputs


def run_detect_motion(impl, config, arrays):
    """Run a detect_motion over `arrays` in order, returning one mask each.

    Order matters: three-frame differencing carries the previous frames.
    """
    from kwiver.vital.algo import DetectMotion
    from kwiver.vital.types import Image, ImageContainer, Timestamp

    algorithm = DetectMotion.create(impl)

    if algorithm is None:
        raise RuntimeError("detect_motion '{}' is not registered".format(impl))

    _configure(algorithm, config)

    outputs = []
    for index, array in enumerate(arrays):
        timestamp = Timestamp()
        timestamp.set_frame(index)
        timestamp.set_time_seconds(index / 30.0)
        result = algorithm.process_image(
            timestamp, ImageContainer(Image(np.ascontiguousarray(array))),
            False)
        outputs.append(np.array(result.image().asarray(), copy=True))

    return outputs


def describe_detections(detections):
    """A detected_object_set as plain values, ordered as the set is.

    Recorded rather than the drawn image, because a box moving by a pixel is
    what a golden should say, not a few thousand changed pixels.
    """
    out = []

    for detection in detections:
        box = detection.bounding_box
        entry = {
            "bbox": [box.min_x(), box.min_y(), box.max_x(), box.max_y()],
            "confidence": detection.confidence,
        }

        detected_type = detection.type
        names = list(detected_type.class_names()) if detected_type else []
        if names:
            entry["types"] = {name: detected_type.score(name)
                              for name in sorted(names)}

        out.append(entry)

    return out


def run_image_object_detector(impl, config, arrays):
    """Detect on each array, returning one list of detections per input."""
    from kwiver.vital.algo import ImageObjectDetector
    from kwiver.vital.types import Image, ImageContainer

    algorithm = ImageObjectDetector.create(impl)

    if algorithm is None:
        raise RuntimeError(
            "image_object_detector '{}' is not registered".format(impl))

    _configure(algorithm, config)

    return [describe_detections(
                algorithm.detect(
                    ImageContainer(Image(np.ascontiguousarray(array)))))
            for array in arrays]


def run_image_io_save_load(impl, config, arrays, extension, work_dir):
    """Write each array out through the image_io and read it back.

    A decode case says the reader agrees with the recording; this says the
    writer and the reader still agree with each other, which is what catches
    a writer that changes channel order or bit depth on the way out.
    """
    from kwiver.vital.algo import ImageIO
    from kwiver.vital.types import Image, ImageContainer

    algorithm = ImageIO.create(impl)

    if algorithm is None:
        raise RuntimeError("image_io '{}' is not registered".format(impl))

    _configure(algorithm, config)

    outputs = []
    for index, array in enumerate(arrays):
        path = os.path.join(work_dir, "roundtrip_{}{}".format(index, extension))
        algorithm.save(path,
                       ImageContainer(Image(np.ascontiguousarray(array))))
        outputs.append(
            np.array(algorithm.load(path).image().asarray(), copy=True))

    return outputs


def run_image_io_load(impl, config, paths):
    """Read each path with the named image_io, returning the decoded arrays."""
    from kwiver.vital.algo import ImageIO

    algorithm = ImageIO.create(impl)

    if algorithm is None:
        raise RuntimeError("image_io '{}' is not registered".format(impl))

    _configure(algorithm, config)

    return [np.array(algorithm.load(str(path)).image().asarray(), copy=True)
            for path in paths]
