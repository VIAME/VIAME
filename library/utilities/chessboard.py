# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Finding the inner corners of a chessboard calibration target.

What `cv::findChessboardCorners` did, by the same route: threshold the image
against a local mean, trace the dark squares, keep the ones that are
convex quadrilaterals of a sensible size, and then work out how they are
arranged.

The arrangement is the part worth explaining. A board with `columns` by
`rows` **inner** corners is made of `(columns + 1)` by `(rows + 1)` squares,
half of them dark. Two dark squares never share an edge -- they meet at a
single point, diagonally -- and that meeting point is an inner corner. So
the dark squares form their own lattice, turned forty five degrees, and
every link in it is exactly one corner of the answer.

That gives the ordering for free: label the squares by their position in the
diagonal lattice, and each link between two of them names the corner between
them and where it sits in the grid. No angles, no sorting by coordinate, and
nothing that falls apart when the board is photographed at a slant.

The corners come back in the row-major order `cv::findChessboardCorners`
uses, refined with `image_kernels.corner_subpix` -- OpenCV left that to a
separate call, and every caller made it.

Which corner is called first is a choice rather than a fact: a chessboard has
no top, and the lattice above has no idea which of its two axes is the board's
width. Two things are guaranteed instead. The labelling is always a genuine
**turn** of the board and never a mirror, because a mirrored corner list
matches no position of a board in front of a camera and quietly fits a
calibration to a reflected pose. And among the turns, the one whose first
corner is nearest the image origin is taken every time, so two pictures of the
same board agree, which is the only thing a calibration asks. OpenCV settles
the same ambiguity its own way and is no more canonical than this: on a square
board, where a quarter turn also fits, it gives different answers to different
pictures of one target.
"""

import numpy as np

from viame import image_kernels


# Thresholding is tried at several block sizes: a board photographed with a
# gradient across it has no single one that works, which is the whole reason
# `cv::findChessboardCorners` sweeps them too.
BLOCK_SIZES = (0, 11, 21, 31, 41)

# How far above the local mean a pixel has to be to count as light.
THRESHOLD_OFFSET = 5.0

# A traced contour is kept as a square if it is convex, has four corners, and
# covers between these fractions of the image.
MIN_AREA_FRACTION = 2.0e-5
MAX_AREA_FRACTION = 0.25

# Two squares are taken to meet when their nearest corners are within this
# fraction of the smaller square's size.
MEETING_TOLERANCE = 0.30

# A corner one step beyond a grid's edge counts as really being there when a
# traced corner sits within this fraction of the grid's spacing of it.
CONTINUATION_TOLERANCE = 0.30


def _binarise(gray, block):
    """Light pixels as a mask, thresholded against a local mean.

    `cv2.adaptiveThreshold` with `ADAPTIVE_THRESH_MEAN_C`. A block of zero
    means a single global threshold instead, which is what a cleanly lit
    synthetic board wants and what a local mean handles worst -- over a large
    flat region the mean sits in the middle of the noise and the result is
    salt and pepper.
    """
    if block <= 0:
        return (gray.astype(np.float64) >
                gray.astype(np.float64).mean()).astype(np.uint8)

    local = image_kernels.box_blur(gray, block).astype(np.float64)
    return (gray.astype(np.float64) > local - THRESHOLD_OFFSET).astype(np.uint8)


def _quadrilaterals(mask, area_range):
    """The convex four-cornered blobs of `mask`, as arrays of four points.

    The mask is **eroded** first, and that step is not optional: the squares
    of a chessboard meet at their corners, so they are eight-connected, and
    border following traces the whole board as a single blob without it.
    `cv::findChessboardCorners` dilates the other polarity for the same
    reason. Eroding pulls each square back from its corners by a pixel or
    two, which the meeting tolerance below absorbs and the sub-pixel
    refinement afterwards removes.
    """
    mask = image_kernels.erode(np.ascontiguousarray(mask), "rect", 3, 3)

    out = []

    for contour in image_kernels.find_contours(mask):
        if len(contour) < 5:
            continue

        area = image_kernels.contour_area(contour)
        if not area_range[0] <= area <= area_range[1]:
            continue

        # Simplify until it is a quadrilateral, or give up. The tolerance
        # starts small so a genuinely four-sided blob keeps its corners, and
        # grows so a ragged one still reduces.
        perimeter = _perimeter(contour)
        quad = None
        for fraction in (0.02, 0.04, 0.06, 0.09, 0.13):
            candidate = image_kernels.approx_poly(contour,
                                                  fraction * perimeter)
            if len(candidate) == 4:
                quad = candidate
                break

        if quad is None or not _is_convex(quad):
            continue

        # A square seen at a slant is still roughly as long as it is wide;
        # a sliver is a mis-trace.
        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        if sides.min() <= 0 or sides.max() / sides.min() > 4.0:
            continue

        out.append(quad)

    return out


def _perimeter(contour):
    return float(np.linalg.norm(np.roll(contour, -1, axis=0) - contour,
                                axis=1).sum())


def _is_convex(quad):
    """Every turn the same way round.

    Written out rather than through `np.cross`, which deprecated
    two-dimensional vectors in numpy 2.0 and warns once per quadrilateral --
    several hundred times over a single board.
    """
    edges = np.roll(quad, -1, axis=0) - quad
    following = np.roll(edges, -1, axis=0)

    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]

    return bool(np.all(crosses > 0.0) or np.all(crosses < 0.0))


def _links(quads):
    """Which squares meet which, and at what point.

    Returns `{(a, b): (corner_a, corner_b, point)}` for each pair that meets,
    where the corner indices say which vertex of each square it was. Two dark
    squares meet at one point only, so a pair with more than one candidate is
    dropped rather than guessed at.
    """
    sizes = [np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1).mean()
             for q in quads]

    found = {}
    for a in range(len(quads)):
        for b in range(a + 1, len(quads)):
            tolerance = MEETING_TOLERANCE * min(sizes[a], sizes[b])

            gaps = np.linalg.norm(quads[a][:, None, :] - quads[b][None, :, :],
                                  axis=2)
            close = np.argwhere(gaps <= tolerance)

            if len(close) != 1:
                continue

            ia, ib = int(close[0][0]), int(close[0][1])
            found[(a, b)] = (ia, ib, (quads[a][ia] + quads[b][ib]) / 2.0)

    return found


def _lattice(quads, links):
    """Place each square on the diagonal lattice, by walking the links.

    A square's four corners run around it, so its two neighbours through
    opposite corners lie in opposite directions. Following corner `i` from
    one square and arriving at corner `j` of the next fixes the step taken,
    and every square reached inherits it. The walk starts from the
    best-connected square, which is the one most likely to be in the middle.
    """
    neighbours = {}
    for (a, b), (ia, ib, _) in links.items():
        neighbours.setdefault(a, []).append((b, ia, ib))
        neighbours.setdefault(b, []).append((a, ib, ia))

    if not neighbours:
        return {}

    start = max(neighbours, key=lambda q: len(neighbours[q]))

    # The step each corner of the starting square walks in. Corners run
    # around the quad, so corner 0 and corner 2 are opposite and must step
    # opposite ways; the same for 1 and 3.
    steps = {0: (-1, -1), 1: (1, -1), 2: (1, 1), 3: (-1, 1)}

    placed = {start: (0, 0)}
    # Each square also carries how its own corner indices map onto those
    # steps, because a neighbour may be wound the other way round.
    rotation = {start: 0}

    pending = [start]
    while pending:
        here = pending.pop()
        row, column = placed[here]
        turn = rotation[here]

        for other, mine, theirs in neighbours.get(here, ()):
            if other in placed:
                continue

            step = steps[(mine - turn) % 4]
            placed[other] = (row + step[1], column + step[0])

            # Arriving at corner `theirs`, the neighbour is entered from the
            # direction opposite the one we left by, which fixes its winding.
            rotation[other] = (theirs - ((mine - turn) % 4) - 2) % 4
            pending.append(other)

    return placed


def _corners_from_lattice(quads, links, placed):
    """Every meeting point, with the grid position it sits at.

    A link between squares at `(r, c)` and `(r + dr, c + dc)` crosses at the
    corner between them, whose grid position is the mean of the two -- in
    half steps, which come out whole once the whole board is offset.
    """
    corners = {}

    for (a, b), (_, _, point) in links.items():
        if a not in placed or b not in placed:
            continue

        (ra, ca), (rb, cb) = placed[a], placed[b]

        if abs(ra - rb) != 1 or abs(ca - cb) != 1:
            continue

        key = (ra + rb, ca + cb)
        corners.setdefault(key, []).append(point)

    return {k: np.mean(v, axis=0) for k, v in corners.items()}


def _grid(corners, columns, rows):
    """A completely filled `rows` by `columns` block of corners, or None.

    The lattice positions are in half steps and start wherever the walk
    began, so this looks for a block of the right size that is entirely
    present.

    **Both orientations are tried.** Which of the lattice's two axes is the
    board's width is arbitrary -- the walk has no idea -- so a board comes
    out the long way round about half the time, and looking for the asked-for
    shape alone finds nothing on those. That is not a rare corner: it cost
    two boards in ten before it was handled. The block keeps whatever shape
    it was found in, and `_orient` turns it the right way up; transposing it
    here instead mirrors the board, which is the one thing the ordering must
    never do.

    A board with a square missed is not silently returned short:
    `cv::findChessboardCorners` is all or nothing and so is this.
    """
    if len(corners) < columns * rows:
        return None

    keys = np.array(list(corners))
    row_values = sorted(set(keys[:, 0]))
    column_values = sorted(set(keys[:, 1]))

    shapes = [(rows, columns)]
    if columns != rows:
        shapes.append((columns, rows))

    for down, across in shapes:
        if len(row_values) < down or len(column_values) < across:
            continue

        for top in range(len(row_values) - down + 1):
            for left in range(len(column_values) - across + 1):
                wanted = [(row_values[top + j], column_values[left + i])
                          for j in range(down) for i in range(across)]

                if not all(key in corners for key in wanted):
                    continue

                if not _is_maximal(corners, row_values, column_values,
                                   top, left, down, across):
                    continue

                return np.array(
                    [corners[key] for key in wanted]).reshape(down, across, 2)

    return None


def _is_maximal(corners, row_values, column_values, top, left, down, across):
    """Whether the block is the whole board rather than part of a bigger one.

    A smaller grid inside a chessboard is also a grid: ask a seven by five
    board for six by five and there are two ways to lay it out, both of them
    real. `cv::findChessboardCorners` refuses those, and it is right to --
    a board is a physical object, and a caller pairing corners against a
    fixed set of object points gets a silently wrong calibration from a
    sub-grid rather than an error. So a block with a complete further row or
    column against any of its four sides is rejected here, and the search
    goes on.

    The cost is a board with a spurious extra quadrilateral neatly in line
    against one edge, which would be refused. That wants a false corner in
    exactly the right place to extend the lattice, and refusing is the safe
    way to be wrong.
    """
    def complete_row(index):
        if index < 0 or index >= len(row_values):
            return False

        return all((row_values[index], column_values[left + i]) in corners
                   for i in range(across))

    def complete_column(index):
        if index < 0 or index >= len(column_values):
            return False

        return all((row_values[top + j], column_values[index]) in corners
                   for j in range(down))

    return not (complete_row(top - 1) or complete_row(top + down) or
                complete_column(left - 1) or complete_column(left + across))


def _continues_beyond(block, traced, spacing):
    """Whether the board carries on past one of the block's four edges.

    `_is_maximal` asks the question inside one lattice, and that is not
    always enough. The light squares of a board touch the margin around it
    and get traced as one blob with it, so the light pass sees only the
    *interior* light squares -- on the seven by five fixture, twelve of them,
    forming a complete and perfectly maximal five by three lattice of their
    own. Every local test passes: it is a real grid, with real squares all
    round it. Only the rest of the picture says it is the middle of a bigger
    board.

    So the rest of the picture is what is asked. Step one place out from each
    edge, using the grid's own spacing, and ask what meets the image there.
    The test cannot simply be "a traced corner", because one step beyond a
    real board is its outer rim, and the rim is made of corners too. What
    separates them is **colour**: inside a board, every corner is where two
    dark squares and two light ones meet, so both passes have a quadrilateral
    touching it. On the rim only one colour is there, the other side being
    the margin. So an edge with squares of both colours the whole way past it
    has more board beyond, and an edge without has ended.
    """
    down, across = block.shape[0], block.shape[1]

    edges = (
        [2.0 * block[j, 0] - block[j, 1] for j in range(down)],
        [2.0 * block[j, across - 1] - block[j, across - 2]
         for j in range(down)],
        [2.0 * block[0, i] - block[1, i] for i in range(across)],
        [2.0 * block[down - 1, i] - block[down - 2, i] for i in range(across)],
    )

    near = CONTINUATION_TOLERANCE * spacing

    corners_of = [np.array([point for quad in quads for point in quad])
                  if quads else np.zeros((0, 2))
                  for quads in traced]

    for beyond in edges:
        outside = np.asarray(beyond)

        both = np.ones(len(outside), dtype=bool)

        for vertices in corners_of:
            if not len(vertices):
                both[:] = False
                break

            gaps = np.linalg.norm(outside[:, None, :] - vertices[None, :, :],
                                  axis=2).min(axis=1)
            both &= gaps <= near

        if np.all(both):
            return True

    return False


def _relabellings(block):
    """The eight ways a grid of corners can be labelled.

    The four turns of the board, and each of those read backwards along its
    rows. Four are turns and four are mirrors, and `_orient` keeps only the
    turns.
    """
    for turned in (block, np.rot90(block, 1), np.rot90(block, 2),
                   np.rot90(block, 3)):
        yield turned
        yield turned[:, ::-1]


def _handedness(block):
    """Positive when the block runs the same way round as a board in view.

    Across a row and then down a column are two directions in the image, and
    a board photographed from the front always turns from the first to the
    second the same way. A labelling that turns the other way is the board
    mirrored -- it still looks like a tidy grid, and it is not one, because no
    position of a board in front of a camera produces it. Calibrating against
    it fits a pose reflected through the image plane, so it has to be
    rejected here rather than puzzled over later.
    """
    across = block[0, -1] - block[0, 0]
    down = block[-1, 0] - block[0, 0]

    return float(across[0] * down[1] - across[1] * down[0])


def _orient(block, columns, rows):
    """`cv::findChessboardCorners`' ordering: along a row, then down.

    Of the eight ways to label the block, the mirrored four are dropped
    outright and the wrong-shaped ones with them; which of the rest is meant
    is genuinely unknowable -- a chessboard has no top -- so the board is
    turned until its first corner is the one nearest the image origin. That
    is a rule, not a recovery of the truth, and it is the same rule for every
    view, which is all a calibration asks: that the corner called first is
    the same physical corner in each picture.

    OpenCV settles the same ambiguity its own way and is no more canonical;
    on a square board, where a quarter turn also fits, it gives different
    answers to different pictures of the same target.
    """
    best = None

    for option in _relabellings(block):
        if option.shape[0] != rows or option.shape[1] != columns:
            continue

        if _handedness(option) <= 0.0:
            continue

        distance = float(np.linalg.norm(option[0, 0]))

        if best is None or distance < best[0]:
            best = (distance, option)

    if best is None:
        return None

    return np.ascontiguousarray(best[1].reshape(-1, 2))


def find_chessboard_corners(gray, columns, rows, refine=True):
    """The inner corners of a chessboard, or `(False, None)`.

    `cv2.findChessboardCorners`, with `cornerSubPix` folded in -- OpenCV left
    the refinement to a separate call and every caller in this tree made it.

    @param gray a single plane image of the board
    @param columns inner corners across
    @param rows inner corners down
    @param refine whether to refine to sub-pixel accuracy

    Returns `(found, corners)` with the corners as a `columns * rows` by 2
    array in row-major order.
    """
    gray = np.ascontiguousarray(gray)

    if gray.ndim != 2:
        raise ValueError("find_chessboard_corners wants a single plane image")

    if columns < 2 or rows < 2:
        raise ValueError("a chessboard needs at least two corners each way")

    pixels = float(gray.shape[0] * gray.shape[1])
    area_range = (MIN_AREA_FRACTION * pixels, MAX_AREA_FRACTION * pixels)

    for block in BLOCK_SIZES:
        light = _binarise(gray, block)

        # Both polarities: which of the two colours is "dark" depends on the
        # board and the lighting, and only one of them tiles the way the
        # lattice walk expects. Both are traced before either is used,
        # because `_continues_beyond` needs the corners one pass found to
        # judge what the other pass is looking at.
        traced = [_quadrilaterals(np.ascontiguousarray(mask), area_range)
                  for mask in (1 - light, light)]

        for quads in traced:
            if len(quads) < (columns * rows) // 4:
                continue

            links = _links(quads)
            if not links:
                continue

            placed = _lattice(quads, links)
            corners = _corners_from_lattice(quads, links, placed)
            grid = _grid(corners, columns, rows)

            if grid is None:
                continue

            ordered = _orient(grid, columns, rows)

            if ordered is None:
                continue

            laid_out = ordered.reshape(rows, columns, 2)
            spacing = float(np.median(np.linalg.norm(
                np.diff(laid_out, axis=1), axis=2)))

            if _continues_beyond(laid_out, traced, spacing):
                continue

            if refine:
                ordered = image_kernels.corner_subpix(
                    gray, ordered, 5, 5, 30, 0.001)

            return True, ordered

    return False, None
