"""Input images the measurement golden needs.

The 96 by 64 fixtures the filters are recorded on are too small for any of
this: a stereo matcher with 128 disparities needs a wider image than that,
and a calibration target needs room for a grid. So these are their own size,
and larger.

Written into `inputs/` beside the others, generated once and committed.
"""

import numpy as np

SEED = 20260911

# Wide enough for a disparity search, and a shape no other fixture has so
# that a case reading the wrong one fails on its shape rather than quietly.
STEREO_WIDTH = 320
STEREO_HEIGHT = 240

# The default `target_width` by `target_height` of
# `ocv_detect_calibration_targets` is 7 by 5 inner corners, which is an 8 by
# 6 board.
BOARD_COLUMNS = 8
BOARD_ROWS = 6
SQUARE = 40
BOARD_MARGIN = 40


def _texture(rng, width, height):
    """A background with structure at several scales.

    A stereo matcher needs texture to match on: a smooth gradient gives it
    nothing and it returns its invalid value everywhere, which is a golden
    that says nothing. This is noise smoothed by summation at three scales,
    which is textured at every block size the matcher might use.
    """
    image = np.zeros((height, width), dtype=np.float64)

    for scale, weight in ((4, 1.0), (16, 0.7), (64, 0.4)):
        coarse = rng.random((height // scale + 2, width // scale + 2))
        enlarged = np.kron(coarse, np.ones((scale, scale)))
        image += weight * enlarged[:height, :width]

    image -= image.min()
    image /= max(image.max(), 1e-9)

    return image


# The two disparities the stereo fixture is built at. Whole pixels, so the
# truth is exact: a sub-pixel shift would need resampling and the recording
# would then be measuring the resampler too.
BACKGROUND_DISPARITY = 8
FOREGROUND_DISPARITY = 40

# Where the nearer patch sits in the left image.
PATCH_TOP = 70
PATCH_BOTTOM = 170
PATCH_LEFT = 110
PATCH_RIGHT = 230


def stereo_pair(rng):
    """A textured background and a nearer patch, at two known disparities.

    A point at column x in the left image is at column x - d in the right,
    so the background is built by taking the right image further along the
    same texture and the patch by placing one piece of texture at two
    columns that differ by its disparity. The depth map then has two plateaus
    a matcher either finds or does not.

    The texture matters as much as the geometry: a stereo matcher given a
    smooth gradient returns its invalid value everywhere, and a recording of
    that says nothing about a replacement.
    """
    background = BACKGROUND_DISPARITY
    foreground = FOREGROUND_DISPARITY

    base = _texture(rng, STEREO_WIDTH + foreground, STEREO_HEIGHT)

    left = base[:, :STEREO_WIDTH].copy()
    right = base[:, background:background + STEREO_WIDTH].copy()

    # A patch of its own texture, so the matcher has something to lock onto
    # at a disparity the background does not have.
    height = PATCH_BOTTOM - PATCH_TOP
    width = PATCH_RIGHT - PATCH_LEFT
    patch = _texture(rng, width, height)

    left[PATCH_TOP:PATCH_BOTTOM, PATCH_LEFT:PATCH_RIGHT] = patch
    right[PATCH_TOP:PATCH_BOTTOM,
          PATCH_LEFT - foreground:PATCH_RIGHT - foreground] = patch

    def to_rgb(plane):
        bytes_ = np.clip(plane * 235 + 10, 0, 255).astype(np.uint8)
        return np.dstack([bytes_] * 3)

    return to_rgb(left), to_rgb(right)


def chessboard(rng):
    """An 8 by 6 chessboard, which is 7 by 5 inner corners.

    Rendered exactly rather than photographed: a synthetic board makes the
    corner positions known to the pixel, so a recording of where the detector
    put them is a recording of its sub-pixel refinement rather than of the
    fixture's own blur.
    """
    width = BOARD_COLUMNS * SQUARE + 2 * BOARD_MARGIN
    height = BOARD_ROWS * SQUARE + 2 * BOARD_MARGIN

    image = np.full((height, width), 230, dtype=np.uint8)

    for row in range(BOARD_ROWS):
        for column in range(BOARD_COLUMNS):
            if (row + column) % 2:
                continue

            y = BOARD_MARGIN + row * SQUARE
            x = BOARD_MARGIN + column * SQUARE
            image[y:y + SQUARE, x:x + SQUARE] = 25

    # A little noise, so a detector that only works on a perfectly clean
    # image is not what is being recorded.
    image = image.astype(np.int16) + rng.integers(-4, 5, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)

    return np.dstack([image] * 3)


def dot_grid(rng):
    """A 7 by 5 grid of filled circles, the other target type.

    `ocv_detect_calibration_targets` detects dot boards with a blob detector
    whose area and circularity limits are configurable, so the dots are well
    inside the shipped `dot_min_area` and `dot_max_area`.
    """
    columns, rows = 7, 5
    spacing = 50
    radius = 12
    margin = 45

    width = (columns - 1) * spacing + 2 * margin
    height = (rows - 1) * spacing + 2 * margin

    image = np.full((height, width), 235, dtype=np.uint8)
    ys, xs = np.mgrid[0:height, 0:width]

    for row in range(rows):
        for column in range(columns):
            centre_y = margin + row * spacing
            centre_x = margin + column * spacing
            distance = np.sqrt((xs - centre_x) ** 2 + (ys - centre_y) ** 2)
            image[distance <= radius] = 20

    image = image.astype(np.int16) + rng.integers(-3, 4, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)

    return np.dstack([image] * 3)


def build():
    """Return {name: image} for the measurement group's own fixtures."""
    rng = np.random.default_rng(SEED)

    left, right = stereo_pair(rng)

    return {
        "stereo_left": left,
        "stereo_right": right,
        "chessboard": chessboard(rng),
        "dot_grid": dot_grid(rng),
    }
