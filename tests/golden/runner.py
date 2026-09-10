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
