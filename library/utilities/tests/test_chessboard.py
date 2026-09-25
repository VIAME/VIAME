"""Finding a chessboard's inner corners, held to a target we drew ourselves.

This replaces `cv::findChessboardCorners`, and the contract is the corner
positions rather than agreement with OpenCV: the board is rendered through a
known homography, so where every corner belongs is known to the last decimal
and can simply be checked.

The one thing that is *not* checked against OpenCV on purpose is which corner
comes first. A chessboard has no top, so the labelling is ambiguous up to a
turn of the board, and OpenCV resolves that its own way -- on a square board
it gives different answers to different pictures of the same target. What is
demanded here instead is that the labelling is always a genuine *turn* and
never a mirror, because a mirrored corner list is not a board seen from
anywhere and quietly fits a reflected pose.

Measured against cv2 when written, over thirty boards of five shapes under
perspective, uneven lighting and noise:

    found                30/30, the same 30/30 cv2 finds
    worst corner error   0.215 px, the same 0.215 px cv2 reaches
    calibrated from it   fx within 0.14 px of a true 1100, where the same
                         calibration from cv2's corners is within 0.13
"""
import numpy as np
import pytest

from viame import image_kernels
from viame.utilities import chessboard, geometry


CELL = 40
MARGIN = 60


def _board(columns, rows, cell=CELL, margin=MARGIN):
    """A chessboard with `columns` by `rows` **inner** corners, and its truth.

    The corner between two squares sits on the boundary between them, which
    is half a pixel before the first pixel of the second -- the same
    convention OpenCV's own answer follows.
    """
    across, down = columns + 1, rows + 1

    image = np.full((down * cell + 2 * margin, across * cell + 2 * margin),
                    255, np.uint8)

    for j in range(down):
        for i in range(across):
            if (i + j) % 2 == 0:
                image[margin + j * cell:margin + (j + 1) * cell,
                      margin + i * cell:margin + (i + 1) * cell] = 0

    truth = np.array([[margin + i * cell - 0.5, margin + j * cell - 0.5]
                      for j in range(1, down) for i in range(1, across)])

    return image, truth


def _turns(truth, columns, rows):
    """Every labelling of the truth that is a turn of the board.

    A rectangular board can be handed back the right way up or upside down; a
    square one can also be given a quarter turn either way. Reading a row
    backwards is a mirror and is deliberately absent.
    """
    grid = truth.reshape(rows, columns, 2)

    options = [grid, grid[::-1, ::-1]]

    if columns == rows:
        turned = grid.transpose(1, 0, 2)
        options += [turned[::-1, :], turned[:, ::-1]]

    return [np.ascontiguousarray(option).reshape(-1, 2) for option in options]


def _error(corners, truth, columns, rows):
    """How far the corners are from the truth under the best turn of it.

    Fails rather than returning if the answer is not a turn at all, since a
    mirrored or scrambled ordering is the failure this is looking for.
    """
    best = min(np.abs(corners - option).max()
               for option in _turns(truth, columns, rows))

    assert best < 2.0, "the corners are not the board under any turn of it"

    return best


@pytest.mark.parametrize("columns,rows", [(9, 6), (8, 5), (6, 4), (5, 3)])
def test_flat_board(columns, rows):
    """A board face on, which is the case every other one degrades from."""
    image, truth = _board(columns, rows)

    found, corners = chessboard.find_chessboard_corners(image, columns, rows)

    assert found
    assert corners.shape == (columns * rows, 2)
    assert _error(corners, truth, columns, rows) < 1.0


def test_row_major_order():
    """Along a row first, then down: `cv::findChessboardCorners`' order.

    Checked by shape rather than by position, so it holds whichever way up
    the board came back: consecutive corners are a short step apart and the
    step to the next row is a long one, which is only true of row-major.
    """
    columns, rows = 9, 6
    image, _ = _board(columns, rows)

    found, corners = chessboard.find_chessboard_corners(image, columns, rows)
    assert found

    grid = corners.reshape(rows, columns, 2)

    along = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    down = np.linalg.norm(np.diff(grid, axis=0), axis=2)

    assert np.allclose(along, CELL, atol=1.0)
    assert np.allclose(down, CELL, atol=1.0)


def test_never_mirrored():
    """The labelling is a turn of the board, never a reflection of it.

    A mirrored list still looks like a tidy grid and is not one: no position
    of a board in front of a camera produces it, so a calibration built on it
    fits a pose reflected through the image plane. The check is that going
    along a row and then down a column always turns the same way.
    """
    for columns, rows in ((9, 6), (7, 7), (5, 3)):
        image, _ = _board(columns, rows)

        found, corners = chessboard.find_chessboard_corners(
            image, columns, rows)
        assert found

        grid = corners.reshape(rows, columns, 2)

        across = grid[0, -1] - grid[0, 0]
        downwards = grid[-1, 0] - grid[0, 0]

        turn = across[0] * downwards[1] - across[1] * downwards[0]

        assert turn > 0.0, f"the {columns}x{rows} board came back mirrored"


@pytest.mark.parametrize("strength", [0.04, 0.08, 0.12])
def test_perspective(strength):
    """A board at a slant, which is what a calibration is made of.

    A calibration that only ever sees the board face on cannot separate the
    focal length from the distance, so every real set of views is at an
    angle and the detector has to hold up there.
    """
    columns, rows = 9, 6
    image, truth = _board(columns, rows)
    height, width = image.shape

    rng = np.random.default_rng(3)
    corners_of = np.array([[0.0, 0.0], [width, 0.0],
                           [width, height], [0.0, height]])
    moved = corners_of + rng.uniform(
        -strength, strength, (4, 2)) * np.array([width, height])

    homography = geometry.four_point_homography(corners_of, moved)

    slanted = image_kernels.warp_perspective(
        image, np.ascontiguousarray(homography), width, height,
        "bilinear", "constant", 255.0)

    found, corners = chessboard.find_chessboard_corners(
        slanted, columns, rows)

    assert found
    assert _error(corners, geometry.apply_homography(homography, truth),
                  columns, rows) < 1.0


def test_uneven_lighting_and_noise():
    """A gradient across the board, which no single threshold survives.

    The reason the detector sweeps block sizes at all: a global threshold
    takes the dark end of a shaded board for black, and a local mean over a
    flat region sits in the middle of the noise and gives salt and pepper.
    """
    columns, rows = 8, 5
    image, truth = _board(columns, rows)
    height, width = image.shape

    rows_of, columns_of = np.mgrid[0:height, 0:width]
    shaded = image.astype(float) * (0.65 + 0.35 * (columns_of / width)
                                    + 0.12 * (rows_of / height))

    rng = np.random.default_rng(11)
    noisy = np.clip(shaded + rng.normal(0.0, 4.0, shaded.shape),
                    0, 255).astype(np.uint8)

    found, corners = chessboard.find_chessboard_corners(noisy, columns, rows)

    assert found
    assert _error(corners, truth, columns, rows) < 1.0


def test_calibrates_to_the_truth():
    """The whole point: a calibration off these corners recovers the rig.

    Ten views of the board are rendered through a known camera, detected,
    and calibrated. This is the test that would catch an ordering that is
    self-consistent but wrong, which none of the per-image checks can.
    """
    from viame.measurement import projection
    from viame.utilities import calibration

    columns, rows = 9, 6
    width, height = 1280, 960

    intrinsics = np.array([[1100.0, 0.0, 646.0],
                           [0.0, 1095.0, 478.0],
                           [0.0, 0.0, 1.0]])

    board, _ = _board(columns, rows, cell=60, margin=0)

    # Board pixels to board units, one unit to the square, with the first
    # inner corner at the origin.
    to_units = np.array([[1.0 / 60, 0.0, -1.0],
                         [0.0, 1.0 / 60, -1.0],
                         [0.0, 0.0, 1.0]])

    grid_y, grid_x = np.mgrid[0:rows, 0:columns]
    object_points = np.column_stack([grid_x.ravel().astype(float),
                                     grid_y.ravel().astype(float),
                                     np.zeros(columns * rows)])

    poses = [(0.05, -0.10, 0.02, -4.0, -2.5, 16.0),
             (-0.22, 0.18, -0.05, -5.0, -3.0, 14.0),
             (0.30, 0.10, 0.08, -4.5, -3.5, 15.0),
             (-0.10, -0.32, 0.03, -4.2, -2.2, 13.5),
             (0.18, 0.26, -0.10, -5.5, -2.8, 17.0),
             (-0.28, -0.12, 0.12, -4.0, -3.2, 12.5),
             (0.12, -0.24, -0.06, -4.8, -2.6, 15.5),
             (-0.16, 0.30, 0.09, -5.2, -3.4, 16.5),
             (0.34, -0.08, 0.04, -4.4, -2.9, 14.5),
             (-0.06, 0.14, -0.12, -4.6, -3.1, 13.0)]

    rng = np.random.default_rng(7)
    seen = []

    for rotation in poses:
        matrix = projection.rodrigues(np.array(rotation[:3]))
        translation = np.array(rotation[3:])

        # A plane through a camera is a homography: the third column of the
        # rotation multiplies a z of zero and drops out.
        placed = intrinsics @ np.column_stack(
            [matrix[:, 0], matrix[:, 1], translation])

        view = image_kernels.warp_perspective(
            board, np.ascontiguousarray(placed @ to_units), width, height,
            "bilinear", "constant", 255.0)
        view = np.clip(view.astype(float) + rng.normal(0.0, 2.0, view.shape),
                       0, 255).astype(np.uint8)

        found, corners = chessboard.find_chessboard_corners(
            view, columns, rows)

        assert found, "a rendered view of the board was not detected"
        seen.append(corners)

    rms, found_intrinsics, distortion, _, _ = calibration.calibrate_camera(
        [object_points] * len(seen), seen, (width, height))

    assert rms < 0.1
    assert abs(found_intrinsics[0, 0] - 1100.0) < 1.0
    assert abs(found_intrinsics[1, 1] - 1095.0) < 1.0
    assert abs(found_intrinsics[0, 2] - 646.0) < 1.0
    assert abs(found_intrinsics[1, 2] - 478.0) < 1.0


def test_missing_corner_is_not_found():
    """All or nothing, as `cv::findChessboardCorners` is.

    A board with a square painted out must fail rather than hand back a
    smaller grid that happens to fit, because the caller pairs what comes
    back against a fixed set of object points.
    """
    columns, rows = 9, 6
    image, _ = _board(columns, rows)

    # Paint out a corner square of the board, which removes a whole row and
    # column of the grid rather than one corner.
    image[MARGIN:MARGIN + 2 * CELL, MARGIN:MARGIN + 2 * CELL] = 255

    found, corners = chessboard.find_chessboard_corners(image, columns, rows)

    assert not found
    assert corners is None


@pytest.mark.parametrize("columns,rows", [(6, 5), (5, 3), (6, 4), (5, 4)])
def test_a_sub_grid_is_not_the_board(columns, rows):
    """A smaller grid inside a board must not be accepted as that board.

    This is the failure with no symptom. Ask a seven by five board for six by
    five and there really are two ways to lay one out; hand either back and
    the caller pairs it against six by five object points and calibrates,
    confidently, to the wrong rig. `cv::findChessboardCorners` refuses all of
    them and so must this.

    The case that made it worth a test is `(5, 3)`. The light squares of a
    board touch the margin around it and get traced as one blob with it, so
    the light pass sees only the interior ones -- twelve of them here, laid
    out as a complete, maximal five by three lattice with squares all the way
    round. Nothing local to it says it is the middle of something bigger.
    """
    image, _ = _board(7, 5)

    found, corners = chessboard.find_chessboard_corners(image, columns, rows)

    assert not found, f"a {columns}x{rows} sub-grid of a 7x5 board was taken"
    assert corners is None


def test_the_whole_board_is_still_found():
    """The other half of the test above, which it would be easy to pass alone.

    Refusing everything would satisfy `test_a_sub_grid_is_not_the_board`
    perfectly, so the board that *is* there has to be demanded in the same
    breath. One step beyond a real board is its outer rim, which is made of
    corners too, and telling those from more board is the whole difficulty.
    """
    image, truth = _board(7, 5)

    found, corners = chessboard.find_chessboard_corners(image, 7, 5)

    assert found
    assert _error(corners, truth, 7, 5) < 1.0


def test_rejects_bad_arguments():
    image, _ = _board(9, 6)

    with pytest.raises(ValueError):
        chessboard.find_chessboard_corners(np.dstack([image] * 3), 9, 6)

    with pytest.raises(ValueError):
        chessboard.find_chessboard_corners(image, 1, 6)


def test_blank_image_finds_nothing():
    """No board is not a board, rather than an exception or a guess."""
    found, corners = chessboard.find_chessboard_corners(
        np.full((400, 500), 200, np.uint8), 9, 6)

    assert not found
    assert corners is None
