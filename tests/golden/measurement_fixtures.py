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


# ----------------------------------------------------------------------------
# A synthetic stereo calibration set
# ----------------------------------------------------------------------------
#
# Twelve views of a chessboard through a stereo rig whose parameters are
# known exactly, rendered rather than photographed. That is what makes a
# calibration testable at all: the recording says what the implementation
# produced, and the ground truth below says whether it was right. A
# photographed set can only offer the first.
#
# The rendering is exact because the target is planar: a plane through a
# pinhole camera is a homography, so each view is one inverse warp of a clean
# board image and nothing is approximated. No lens distortion, so a
# calibration that recovers non-zero distortion coefficients is telling on
# itself.

CALIBRATION_VIEWS = 12
CALIBRATION_IMAGE_WIDTH = 640
CALIBRATION_IMAGE_HEIGHT = 480

# Squares, so the inner corner grid is one less each way: 8 by 5, which is
# forty corners.
#
# Forty rather than the detector's default 7 by 5, and the reason is a defect
# rather than a preference. `optimize_stereo_cameras` checks
# `( tracks.size() / 2 ) % 2` and refuses the calibration when it is
# non-zero, which means it refuses **any board with an odd number of
# corners** -- the default 7 by 5 board has thirty-five, so the shipped
# stereo calibration pipeline cannot calibrate its own default target. The
# check it meant to make is that the tracks divide evenly between the two
# cameras. `design/lite-findings.md` records it; the fixture uses a board
# that works so that the rest of the chain can be recorded at all.
CALIBRATION_COLUMNS = 9
CALIBRATION_ROWS = 6
CALIBRATION_SQUARE_MM = 30.0

# The rig the views are rendered through.
#
# Both principal points are at the image centre, `( width - 1 ) / 2`, and
# that is deliberate. The shipped calibrator fits progressively: it starts
# with the full model and then tries fixing the aspect ratio, then the
# principal point, then each distortion coefficient in turn, keeping every
# constraint that does not worsen the error. On data this clean every
# constraint holds, so the principal point ends up fixed at the image centre
# whatever it really was -- a decentred rig would come back with its focal
# length traded against the offset, 2% out on the fixture this replaced.
# Putting the truth where the constrained model puts it is what lets the
# golden check the whole chain against ground truth rather than only against
# itself; `design/lite-findings.md` records the fitting strategy.
CALIBRATION_CENTRE_X = (CALIBRATION_IMAGE_WIDTH - 1) / 2.0
CALIBRATION_CENTRE_Y = (CALIBRATION_IMAGE_HEIGHT - 1) / 2.0

CALIBRATION_K_LEFT = ((600.0, 0.0, CALIBRATION_CENTRE_X),
                      (0.0, 600.0, CALIBRATION_CENTRE_Y),
                      (0.0, 0.0, 1.0))
CALIBRATION_K_RIGHT = ((610.0, 0.0, CALIBRATION_CENTRE_X),
                       (0.0, 610.0, CALIBRATION_CENTRE_Y),
                       (0.0, 0.0, 1.0))

# Right relative to left: a small rotation and a 120 mm baseline.
CALIBRATION_ROTATION_VECTOR = (0.004, -0.02, 0.001)
CALIBRATION_TRANSLATION_MM = (-120.0, 2.0, 5.0)

# The board image is drawn at this many pixels per millimetre and then warped
# down, so the warp is always minifying and never invents detail.
BOARD_PIXELS_PER_MM = 2
BOARD_MARGIN_MM = 20.0

CALIBRATION_SEED = 4242


def _rodrigues(vector):
    """A rotation matrix from an axis-angle vector, without cv2."""
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))

    if angle == 0.0:
        return np.eye(3)

    axis = vector / angle
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])

    return (np.eye(3) + np.sin(angle) * cross +
            (1.0 - np.cos(angle)) * (cross @ cross))


def _board_image():
    """The calibration board, drawn in millimetre coordinates."""
    width = int((CALIBRATION_COLUMNS * CALIBRATION_SQUARE_MM +
                 2 * BOARD_MARGIN_MM) * BOARD_PIXELS_PER_MM)
    height = int((CALIBRATION_ROWS * CALIBRATION_SQUARE_MM +
                  2 * BOARD_MARGIN_MM) * BOARD_PIXELS_PER_MM)

    image = np.full((height, width), 235, dtype=np.uint8)

    margin = int(BOARD_MARGIN_MM * BOARD_PIXELS_PER_MM)
    square = int(CALIBRATION_SQUARE_MM * BOARD_PIXELS_PER_MM)

    for row in range(CALIBRATION_ROWS):
        for column in range(CALIBRATION_COLUMNS):
            if (row + column) % 2:
                continue

            y = margin + row * square
            x = margin + column * square
            image[y:y + square, x:x + square] = 20

    return image


def calibration_poses():
    """Where the board is, per view, relative to the left camera.

    Spread in all three rotations and in depth, because a calibration from
    views that differ only by translation cannot separate the focal length
    from the distance and comes out confidently wrong.
    """
    poses = []

    for index in range(CALIBRATION_VIEWS):
        rotation = _rodrigues((0.18 * np.sin(index * 1.1),
                               0.22 * np.cos(index * 0.9),
                               0.10 * np.sin(index * 0.5)))

        translation = np.array([
            -CALIBRATION_COLUMNS * CALIBRATION_SQUARE_MM / 2.0
            + 18.0 * np.sin(index * 0.8) + 75.0,
            -CALIBRATION_ROWS * CALIBRATION_SQUARE_MM / 2.0
            + 14.0 * np.cos(index * 0.7),
            470.0 + 55.0 * np.sin(index * 0.6)])

        poses.append((rotation, translation))

    return poses


def _render_view(intrinsics, rotation, translation, board):
    """One view of the board, by inverse warp through the exact homography."""
    intrinsics = np.asarray(intrinsics, dtype=np.float64)

    # Board millimetres to image pixels. A plane is a homography, so the two
    # in-plane columns of the rotation and the translation are all of it.
    homography = intrinsics @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation])

    # Board millimetres to board image pixels.
    offset = BOARD_MARGIN_MM * BOARD_PIXELS_PER_MM
    to_board = np.array([[BOARD_PIXELS_PER_MM, 0.0, offset],
                         [0.0, BOARD_PIXELS_PER_MM, offset],
                         [0.0, 0.0, 1.0]])

    inverse = to_board @ np.linalg.inv(homography)

    ys, xs = np.mgrid[0:CALIBRATION_IMAGE_HEIGHT, 0:CALIBRATION_IMAGE_WIDTH]
    points = np.stack([xs.ravel().astype(np.float64),
                       ys.ravel().astype(np.float64),
                       np.ones(xs.size)])

    mapped = inverse @ points
    board_x = mapped[0] / mapped[2]
    board_y = mapped[1] / mapped[2]

    height, width = board.shape
    out = np.full(xs.size, 255.0)

    inside = ((board_x >= 0) & (board_x <= width - 1) &
              (board_y >= 0) & (board_y <= height - 1) & (mapped[2] > 0))

    x0 = np.floor(board_x[inside]).astype(int)
    y0 = np.floor(board_y[inside]).astype(int)
    fx = board_x[inside] - x0
    fy = board_y[inside] - y0
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    values = board.astype(np.float64)
    out[inside] = (values[y0, x0] * (1 - fx) * (1 - fy) +
                   values[y0, x1] * fx * (1 - fy) +
                   values[y1, x0] * (1 - fx) * fy +
                   values[y1, x1] * fx * fy)

    return np.clip(out.reshape(CALIBRATION_IMAGE_HEIGHT,
                               CALIBRATION_IMAGE_WIDTH), 0, 255).astype(np.uint8)


def calibration_views():
    """`{name: image}` for the synthetic stereo calibration set."""
    board = _board_image()
    right_rotation = _rodrigues(CALIBRATION_ROTATION_VECTOR)
    right_translation = np.array(CALIBRATION_TRANSLATION_MM)

    out = {}

    for index, (rotation, translation) in enumerate(calibration_poses()):
        out["calib_left_%02d" % index] = _render_view(
            CALIBRATION_K_LEFT, rotation, translation, board)

        out["calib_right_%02d" % index] = _render_view(
            CALIBRATION_K_RIGHT,
            right_rotation @ rotation,
            right_rotation @ translation + right_translation,
            board)

    return out


def build():
    """Return {name: image} for the measurement group's own fixtures."""
    rng = np.random.default_rng(SEED)

    left, right = stereo_pair(rng)

    images = {
        "stereo_left": left,
        "stereo_right": right,
        "chessboard": chessboard(rng),
        "dot_grid": dot_grid(rng),
    }
    images.update(calibration_views())

    return images
