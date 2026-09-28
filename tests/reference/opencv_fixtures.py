"""Extra input images the OpenCV golden needs.

The still fixtures in `fixtures.py` were built for the VXL filters, which are
per-pixel and per-neighbourhood: a gradient with bars and a disc is enough to
make them differ. Three of the OpenCV implementations look for structure the
VXL set does not have, and a recording of "found nothing" would say nothing
about a replacement.

Written into `inputs/` beside the others, generated once and committed.
"""

import numpy as np

import fixtures

WIDTH = fixtures.WIDTH
HEIGHT = fixtures.HEIGHT
SEED = fixtures.SEED + 1


def circles_rgb(rng):
    """Clean circles for the Hough detector: four radii, one filled.

    Hough wants gradient edges on a quiet background, which the shared
    fixture's bars and gradient destroy. The radii are spread so that a
    detector using the wrong accumulator resolution loses one of them.

    Three channel, because `hough_circle` converts BGR to gray on the way in
    and refuses a single channel image.
    """
    image = np.full((HEIGHT, WIDTH), 210, dtype=np.uint8)
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]

    for centre_x, centre_y, radius, filled in (
            (22, 22, 11, False),
            (60, 20, 7, True),
            (48, 46, 15, False),
            (84, 46, 5, False)):
        distance = np.sqrt((xs - centre_x) ** 2 + (ys - centre_y) ** 2)
        if filled:
            image[distance <= radius] = 40
        else:
            image[np.abs(distance - radius) <= 1.2] = 40

    image = image.astype(np.int16) + rng.integers(-3, 4, image.shape)
    return np.dstack([np.clip(image, 0, 255).astype(np.uint8)] * 3)


def heat(rng):
    """A single channel heat map with blobs of several areas.

    `detect_heat_map` thresholds, finds connected components and filters them
    by area and fill fraction, so the blobs are sized either side of the
    `min_area` the shipped pipeline uses and one is a ring, whose fill
    fraction is low enough to be dropped.
    """
    image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]

    # Well over min_area, solid
    image[(xs - 20) ** 2 + (ys - 20) ** 2 < 12 ** 2] = 240
    # Just under min_area at 100, solid
    image[(xs - 52) ** 2 + (ys - 14) ** 2 < 5 ** 2] = 200
    # Large but hollow: area passes, fill fraction does not
    ring = np.abs(np.sqrt((xs - 70.0) ** 2 + (ys - 44.0) ** 2) - 14) < 2
    image[ring] = 220
    # A rectangle, so a contour tracer and a bounding box disagree if either
    # is wrong about inclusive edges
    image[8:14, 74:92] = 180

    image = image.astype(np.int16) + rng.integers(0, 6, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def bayer_bg(rng):
    """An 8-bit BG Bayer mosaic of the shared RGB fixture.

    `ocv_debayer` is configured `pattern: BG` by every pipeline that selects
    it. Built by sampling the RGB fixture rather than by generating something
    new, so the demosaic result can be compared against `inputs/rgb8.png`.

    BG means the top-left 2x2 is
        B G
        G R
    """
    rgb = fixtures.rgb8(rng)
    mosaic = np.zeros(rgb.shape[:2], dtype=np.uint8)

    mosaic[0::2, 0::2] = rgb[0::2, 0::2, 2]   # blue
    mosaic[0::2, 1::2] = rgb[0::2, 1::2, 1]   # green
    mosaic[1::2, 0::2] = rgb[1::2, 0::2, 1]   # green
    mosaic[1::2, 1::2] = rgb[1::2, 1::2, 0]   # red

    return mosaic


def build():
    """Return {name: image} for the OpenCV group's own fixtures."""
    rng = np.random.default_rng(SEED)

    return {
        "circles_rgb": circles_rgb(rng),
        "heat": heat(rng),
        "bayer_bg": bayer_bg(rng),
    }
