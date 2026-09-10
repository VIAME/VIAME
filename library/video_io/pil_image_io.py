# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Image reading and writing on Pillow, as the fallback the codecs need.

`library/video_io/codecs` handles PNG, JPEG, BMP and baseline TIFF, which is
every format VIAME's own pipelines read and write. What it deliberately does
not handle is everything else: a tiled TIFF, a 32 bit float TIFF, a WebP or
a GIF that arrives with somebody's survey data. Guessing at those in C++
would mean carrying a second decoder for cases nobody has; handing them to
Pillow means the C++ side stays the size of the problem VIAME actually has.

So this is not a rival implementation, it is the long tail.
`core_image_io` asks `codecs::can_read` first and only comes here when the
answer is no, which is why this registers under a name of its own rather
than aliasing one of the existing readers.

Pillow's own modes are the mapping that matters:

* `L`, `I;16`, `I;16B`, `I;16L` -> one plane, 8 or 16 bit
* `RGB`, `RGBA` -> three or four planes, 8 bit
* `P` (palette), `1` (bilevel), `LA`, `CMYK` and the rest are converted to
  the nearest of the above rather than refused, because a caller that got
  here has already been refused once
* `F` and `I` (32 bit) are converted to 16 bit only when they fit, and
  refused otherwise: silently losing a float range is worse than failing
"""

import logging
import os

import numpy as np

from kwiver.vital.algo import ImageIO
from kwiver.vital.types import Image, ImageContainer, Metadata

logger = logging.getLogger(__name__)


# Pillow mode -> (numpy dtype, plane count), for the modes that need no
# conversion. Everything else goes through `convert` to one of these.
DIRECT_MODES = {
    "L": (np.uint8, 1),
    "I;16": (np.uint16, 1),
    "I;16L": (np.uint16, 1),
    "I;16B": (np.uint16, 1),
    "RGB": (np.uint8, 3),
    "RGBA": (np.uint8, 4),
}

# What a mode Pillow gives us that is not in DIRECT_MODES becomes.
CONVERSIONS = {
    "1": "L",
    "P": "RGB",
    "PA": "RGBA",
    "LA": "RGBA",
    "CMYK": "RGB",
    "YCbCr": "RGB",
    "HSV": "RGB",
    "La": "RGBA",
    "RGBX": "RGB",
    "RGBa": "RGBA",
}


class PILImageIO(ImageIO):
    """Read and write images with Pillow."""

    def __init__(self):
        ImageIO.__init__(self)

    def get_configuration(self):
        return ImageIO.get_configuration(self)

    def set_configuration(self, config):
        pass

    def check_configuration(self, config):
        return True

    def skip_path_validation_(self):
        """Whether the base class should stop checking the path itself.

        False: let it check. The base `load` refuses a path that does not
        exist or is a directory, and `save` refuses one whose directory does
        not exist, which is what every other image_io gets and what the
        callers already handle.
        """
        return False

    # ------------------------------------------------------------------
    def load_(self, filename):
        from PIL import Image as PILImage

        with PILImage.open(filename) as handle:
            mode = handle.mode

            if mode not in DIRECT_MODES:
                target = CONVERSIONS.get(mode)

                if target is None:
                    # 32 bit integer and float. Converting either to 8 bit is
                    # what Pillow would do and is almost never what the data
                    # meant, so say so instead.
                    raise RuntimeError(
                        "{}: Pillow mode '{}' has no lossless mapping to a "
                        "vital image; convert it first".format(filename, mode))

                logger.debug("%s: converting Pillow mode %s to %s",
                             filename, mode, target)
                handle = handle.convert(target)
                mode = target

            array = np.asarray(handle)

        dtype, planes = DIRECT_MODES[mode]
        array = np.ascontiguousarray(array, dtype=dtype)

        # Pillow hands back (h, w) for one plane and (h, w, c) otherwise;
        # vital wants the plane axis either way
        if array.ndim == 2:
            array = array[:, :, np.newaxis]

        if array.shape[2] != planes:
            raise RuntimeError(
                "{}: Pillow mode '{}' gave {} planes, expected {}".format(
                    filename, mode, array.shape[2], planes))

        # The base `load` attaches the metadata; `ImageContainer` has no
        # setter on the python side, and `load_metadata_` is what a caller
        # asking for it goes through anyway.
        return ImageContainer(Image(array))

    # ------------------------------------------------------------------
    def save_(self, filename, data):
        from PIL import Image as PILImage

        array = np.asarray(data.image().asarray())

        # vital carries the plane axis even for one plane; Pillow wants it
        # gone, and its mode then follows from the dtype and the shape
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]

        planes = 1 if array.ndim == 2 else array.shape[2]

        # `fromarray` picks the mode from the dtype -- uint8 2-D is L, uint16
        # 2-D is I;16, three and four planes are RGB and RGBA. Passing `mode`
        # explicitly is deprecated in Pillow 12 and gone in 13, so the check
        # is here rather than there.
        supported = (
            (array.dtype == np.uint8 and planes in (1, 3, 4)) or
            (array.dtype == np.uint16 and planes == 1))

        if not supported:
            raise RuntimeError(
                "{}: Pillow here writes 8 bit images with 1, 3 or 4 planes "
                "and 16 bit gray, not {} with {} planes".format(
                    filename, array.dtype, planes))

        directory = os.path.dirname(filename)
        if directory and not os.path.isdir(directory):
            raise RuntimeError(
                "{}: the directory to write into does not exist".format(
                    filename))

        PILImage.fromarray(np.ascontiguousarray(array)).save(filename)

    # ------------------------------------------------------------------
    def load_metadata_(self, filename):
        from kwiver.vital.types import metadata_tags as tags

        metadata = Metadata()
        metadata.add(tags.tag_traits_by_tag(
            tags.tags.VITAL_META_IMAGE_URI).create_metadata_item(filename))
        return metadata


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        PILImageIO, "pil",
        "Read and write images with Pillow, for the formats the in-house "
        "codecs decline")
