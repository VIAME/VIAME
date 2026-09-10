"""What the codec golden recording covers.

Phase 7 replaces OpenCV's `imread` and `imwrite`. What has to be pinned first
is what the `ocv` image_io decodes each container to and what it writes back,
because every image_io in VIAME is selected by name from a pipeline and a
replacement has to answer the same way.

The containers themselves are `codec_fixtures.py`'s, committed under
`inputs/codecs/`.
"""

import codec_fixtures

# Every container, by fixture name. The decode cases run all of them.
CONTAINERS = tuple(name for name, _ in codec_fixtures.paths(""))

# Where the committed containers live, relative to tests/golden/inputs.
INPUT_SUBDIR = "codecs"


# image_io implementations to record, and the config variants each is asked
# for by a shipped pipeline plus its own defaults.
IMAGE_IO = {
    "ocv": [
        ("defaults", {}),
    ],
}


# Round trip: write each source array out through the image_io and read it
# back with the same one. What this catches that decode alone does not is a
# writer that changes channel order or bit depth on the way out.
#
# JPEG is not round tripped: it is lossy, so a round trip records the encoder
# rather than the container, and the decode cases already cover reading one.
WRITE_EXTENSIONS = (".png", ".bmp", ".tif")

# The fixture arrays the writer is handed, by name in `inputs/`.
WRITE_SOURCES = ("rgb8", "gray8", "gray16")


# Per container, the largest difference a replacement may show. PNG, BMP and
# TIFF are lossless: a decoder that disagrees at all is wrong. JPEG is not,
# and two conforming IDCT implementations differ, so the tolerance there is
# the one the plan states.
TOLERANCES = {
    "__default__": (0.0, 0.0),
    "jpg_gray8": (8.0, 1.0),
    "jpg_rgb8": (8.0, 1.0),
    "jpg_rgb8_444": (8.0, 1.0),
}


# Containers a replacement is allowed not to decode itself, with what is
# expected instead.
FALLBACK = {
    "tiff_rgb8_tiled":
        "a tiled TIFF rather than a stripped one; lite-removals.md 2.2 has "
        "the in-house reader handing this case to the fallback image_io "
        "rather than decoding it. The recording is what the fallback has to "
        "produce",
}


def tolerance(container):
    return TOLERANCES.get(container, TOLERANCES["__default__"])
