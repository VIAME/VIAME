#!/usr/bin/env python3
"""Record what OpenCV computes, so `image_ops` can be checked after it is gone.

The counterpart of `tests/golden/math/record_from_eigen.cxx`: run once, by
hand, while OpenCV is still on the path, and commit the result. The C++ test
`tests/library/image_ops/test_color.cxx` reads it and holds the in-house
kernels to it, at the tolerances `design/tasks/phase-07-drop-opencv.md`
states per function.

    source <install>/setup_viame.sh
    python3 tests/golden/image_ops/record_from_opencv.py

It refuses to overwrite an existing recording without --force, for the same
reason the other recorders do: a golden must not be quietly redefined by the
code it is meant to be checking.

The images are committed as flat integer arrays in JSON rather than as PNGs,
because the C++ side has to read them and a JSON reader is thirty lines where
a PNG decoder is the thing under test. They are cropped rather than whole:
the conversions here are per pixel or three by three, so a 32 by 24 window is
as much evidence as a 96 by 64 one and two orders of magnitude less to carry
in the repository. The crop is taken at an even offset so that a Bayer
mosaic keeps its parity.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import fixtures            # noqa: E402
import opencv_fixtures     # noqa: E402


# The window every case is recorded over: 32 by 24 at (16, 16). Even offsets
# so a Bayer mosaic keeps its parity, and placed over the fixture's disc edge
# and bars so that the window has structure rather than gradient.
CROP_LEFT = 16
CROP_TOP = 16
CROP_WIDTH = 32
CROP_HEIGHT = 24


def crop(array):
    return array[CROP_TOP:CROP_TOP + CROP_HEIGHT,
                 CROP_LEFT:CROP_LEFT + CROP_WIDTH]


def flat(prefix, array):
    """An image as flat `<prefix>_*` keys, so the C++ reader stays trivial.

    Nested objects would need a real parser on the other side; four keys with
    a prefix need none. `data` is in (row, column, plane) order, which is how
    numpy lays it out and how the C++ side walks it.
    """
    if array.ndim == 2:
        array = array[:, :, np.newaxis]

    return {
        prefix + "_width": int(array.shape[1]),
        prefix + "_height": int(array.shape[0]),
        prefix + "_planes": int(array.shape[2]),
        prefix + "_data": [int(v) for v in array.reshape(-1)],
    }


def record():
    import cv2

    rng = np.random.default_rng(fixtures.SEED)
    rgb = fixtures.rgb8(rng)
    gray = fixtures.gray8(rng)
    bayer = opencv_fixtures.build()["bayer_bg"]

    cases = []

    def add(name, source, expected, tolerance, margin=0):
        """One case.

        `tolerance` is the largest per-pixel difference in counts that the
        replacement may show. `margin` is how many pixels of border the
        comparison skips: OpenCV ran on the whole fixture and the recording
        is a window of the result, so a kernel that reads its neighbours sees
        the window's own edge where OpenCV saw real pixels. Only the
        neighbourhood kernels need it, and only by their radius.
        """
        case = {"name": name, "tolerance": tolerance, "margin": margin}
        case.update(flat("input", crop(source)))
        case.update(flat("expected", crop(expected)))
        cases.append(case)

    # Luminance. The same BT.601 fixed point at 14 bits, so the two agree on
    # every pixel but the ones landing exactly on a half -- ten of 768 here,
    # each by one count. OpenCV's own scalar and vector paths disagree there
    # too, so one count is the honest tolerance rather than a concession.
    add("rgb_to_gray", rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1)

    # One plane into three, which is a copy either way
    add("gray_to_rgb", gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), 0)

    # Channel order, which is a permutation and therefore exact
    add("swap_rb", rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0)

    # HSV and Lab go through doubles here and through lookup tables in
    # OpenCV, so they agree to a count rather than exactly. Observed maxima
    # when this was recorded: 1 for both. The tolerances are one above, which
    # leaves room for a different OpenCV build without leaving room for a
    # wrong answer.
    add("rgb_to_hsv", rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), 2)
    add("rgb_to_lab", rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab), 2)

    # The round trips, which say the inverses are inverses. Compared against
    # OpenCV's round trip rather than against the original, so that the two
    # implementations lose the same information in the same places.
    # Observed maxima: 1 for both.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    add("hsv_to_rgb", hsv, cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 2)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    add("lab_to_rgb", lab, cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 2)

    # Demosaic. `COLOR_BayerRG2RGB` rather than `BayerBG2RGB`: OpenCV's
    # Bayer constants name the pattern the other way round, so the one that
    # decodes a blue-at-(0,0) mosaic into RGB is the RG one. See color.h.
    # margin 2: the interpolation reads two pixels out, and this is the only
    # case here that reads anything but the pixel under it. Away from the
    # window's edge the two agree exactly -- observed maximum 0 -- so the
    # plan's "demosaic <= 3" is met with room to spare and the tolerance is
    # 1 rather than 3.
    add("demosaic_bg", bayer, cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB), 1,
        margin=2)

    # ------------------------------------------------------------------
    # filter.h
    #
    # OpenCV's default border for all of these is BORDER_REFLECT_101, which
    # is what `filter.h` defaults to. The margins are the kernel radius: the
    # window's own edge is not the image's, and the recording is a window.
    # ------------------------------------------------------------------

    add("gaussian_blur_3", gray,
        cv2.GaussianBlur(gray, (3, 3), 0), 1, margin=1)
    add("gaussian_blur_5", gray,
        cv2.GaussianBlur(gray, (5, 5), 0), 1, margin=2)
    add("gaussian_blur_5_rgb", rgb,
        cv2.GaussianBlur(rgb, (5, 5), 0), 1, margin=2)
    add("gaussian_blur_5_sigma_2", gray,
        cv2.GaussianBlur(gray, (5, 5), 2.0), 1, margin=2)

    add("box_blur_3", gray, cv2.blur(gray, (3, 3)), 1, margin=1)
    add("box_blur_5", gray, cv2.blur(gray, (5, 5)), 1, margin=2)

    # A gradient has a sign, so these are recorded as CV_16S and shifted by
    # 32768 to survive the unsigned JSON round trip; the C++ side shifts back.
    def signed16(array):
        return (array.astype(np.int32) + 32768).astype(np.uint16)

    add("sobel_dx_3", gray,
        signed16(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)), 1, margin=1)
    add("sobel_dy_3", gray,
        signed16(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)), 1, margin=1)
    add("sobel_dx_5", gray,
        signed16(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=5)), 1, margin=2)
    add("sobel_dxx_3", gray,
        signed16(cv2.Sobel(gray, cv2.CV_16S, 2, 0, ksize=3)), 1, margin=1)

    # filter2D with a kernel that is neither symmetric nor separable, so a
    # flipped kernel or a transposed walk shows up
    corner = np.array([[0.0, -1.0, 0.0],
                       [-1.0, 5.0, -1.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    add("filter2d_sharpen", gray,
        cv2.filter2D(gray, -1, corner,
                     borderType=cv2.BORDER_REFLECT_101), 1, margin=1)

    # The border rules, on a kernel wide enough that the edge dominates.
    #
    # These are the one group that has to run on the *window* rather than on
    # the whole fixture: the point of the case is what happens where the
    # image stops, and on the whole fixture the window's edge is not where it
    # stops. So OpenCV is given the crop and the comparison keeps every
    # pixel -- margin 0 is meant here, unlike everywhere else.
    line = np.ones((1, 9), dtype=np.float64) / 9.0
    window = crop(gray)
    for name, flag in (("replicate", cv2.BORDER_REPLICATE),
                       ("reflect", cv2.BORDER_REFLECT),
                       ("reflect101", cv2.BORDER_REFLECT_101),
                       ("constant", cv2.BORDER_CONSTANT)):
        cases.append(dict(
            {"name": "border_" + name, "tolerance": 1, "margin": 0},
            **flat("input", window),
            **flat("expected",
                   cv2.filter2D(window, -1, line, borderType=flag))))

    add("add_weighted", gray,
        cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (5, 5), 0),
                        -0.5, 0.0), 2, margin=2)

    # ------------------------------------------------------------------
    # warp.h
    #
    # These run on the window rather than on the whole fixture, like the
    # border cases and for the same reason: a resize or a warp of a window is
    # not a window of the resize, so the only way to compare is to give both
    # implementations the same picture. `add_windowed` records the window as
    # the input and OpenCV's result on it as the expectation, at whatever
    # size that result comes out.
    # ------------------------------------------------------------------

    window = crop(gray)
    window_rgb = crop(rgb)

    def add_windowed(name, source, expected, tolerance, margin=0):
        cases.append(dict(
            {"name": name, "tolerance": tolerance, "margin": margin},
            **flat("input", source),
            **flat("expected", expected)))

    # Resize, up and down, in each of the three modes. The sizes are not
    # multiples of the source, so the sample grid has to be right rather than
    # merely consistent.
    for tag, size in (("half", (16, 12)), ("double", (64, 48)),
                      ("odd", (21, 17))):
        # Zero, not one: `resize` reproduces OpenCV's fixed point for an
        # 8-bit image exactly since P7-T08 needed it to. A count of slack
        # was enough for everything before darknet, whose network turns a
        # one-count difference into two fewer detections.
        add_windowed("resize_bilinear_" + tag, window,
                     cv2.resize(window, size,
                                interpolation=cv2.INTER_LINEAR), 0)
        add_windowed("resize_nearest_" + tag, window,
                     cv2.resize(window, size,
                                interpolation=cv2.INTER_NEAREST), 0)
        add_windowed("resize_area_" + tag, window,
                     cv2.resize(window, size,
                                interpolation=cv2.INTER_AREA), 1)

    add_windowed("resize_bilinear_rgb", window_rgb,
                 cv2.resize(window_rgb, (20, 15),
                            interpolation=cv2.INTER_LINEAR), 0)

    # The bilinear warps agree to about four counts rather than one, and the
    # difference is OpenCV's rather than ours: `warpPerspective`,
    # `warpAffine` and `remap` interpolate in fixed point. INTER_BITS is 5,
    # so the fractional position is quantised to a thirty-second of a pixel
    # and the weights to a 2048th, and on a high-contrast edge that costs a
    # few counts. `image_ops` interpolates in double, which is nearer the
    # exact answer -- `test_warp.cxx` checks that claim rather than asserting
    # it -- so matching OpenCV exactly here would mean deliberately
    # reproducing a quantisation that loses accuracy. The tolerance says 5
    # instead, and this comment says why.
    #
    # The nearest-neighbour warps have no interpolation to quantise and are
    # exact.
    WARP_TOLERANCE = 5

    # A perspective warp with real perspective in it, so a homography that is
    # secretly affine cannot pass
    homography = np.array([[1.10, 0.15, -3.0],
                           [-0.08, 0.95, 2.0],
                           [0.0012, 0.0008, 1.0]], dtype=np.float64)
    add_windowed("warp_perspective", window,
                 cv2.warpPerspective(window, homography,
                                     (window.shape[1], window.shape[0]),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0), WARP_TOLERANCE)
    add_windowed("warp_perspective_replicate", window,
                 cv2.warpPerspective(window, homography,
                                     (window.shape[1], window.shape[0]),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE),
                 WARP_TOLERANCE)
    add_windowed("warp_perspective_nearest", window,
                 cv2.warpPerspective(window, homography,
                                     (window.shape[1], window.shape[0]),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0), 0)

    # An affine warp, and the rotation matrix builder that feeds it
    rotation = cv2.getRotationMatrix2D((15.5, 11.5), 20.0, 1.15)
    add_windowed("warp_affine_rotate", window,
                 cv2.warpAffine(window, rotation,
                                (window.shape[1], window.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0), WARP_TOLERANCE)

    # remap, with maps that are neither a resize nor a warp
    ys, xs = np.mgrid[0:window.shape[0], 0:window.shape[1]]
    map_x = (xs + 3.0 * np.sin(ys / 4.0)).astype(np.float32)
    map_y = (ys + 2.0 * np.cos(xs / 5.0)).astype(np.float32)
    add_windowed("remap_wave", window,
                 cv2.remap(window, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0), WARP_TOLERANCE)

    # ------------------------------------------------------------------
    # histogram.h
    #
    # Windowed, like the warps: an equalisation is over the whole picture it
    # is given, so a window of the result is not the result on the window.
    # ------------------------------------------------------------------

    add_windowed("normalize_min_max", window,
                 cv2.normalize(window, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8), 1)

    add_windowed("equalize_hist", window, cv2.equalizeHist(window), 0)

    # An image with a large flat background, where the textbook mapping and
    # OpenCV's differ across the whole range rather than at the ends
    flatish = window.copy()
    flatish[:8, :] = 40
    add_windowed("equalize_hist_flat_background", flatish,
                 cv2.equalizeHist(flatish), 0)

    for tag, clip, tiles in (("clip_3_2x2", 3.0, (2, 2)),
                             ("clip_20_2x2", 20.0, (2, 2)),
                             ("clip_3_4x4", 3.0, (4, 4)),
                             ("clip_40_8x8", 40.0, (8, 8))):
        engine = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
        # CLAHE interpolates between tile mappings and the redistribution of
        # clipped counts is done slightly differently here, so a few counts
        # is what the plan asks for and what this allows.
        add_windowed("clahe_" + tag, window, engine.apply(window), 2)

    # A flat image, where the clipping and the redistribution are the whole
    # answer: every count is in one bin, so what comes out says exactly how
    # the clipped remainder was given back. OpenCV gives 85 at clip 3 with
    # two tiles and 255 at clip 40 with eight, which are different answers to
    # the same question and worth pinning rather than guessing at.
    # `flat` would shadow the `flat()` helper in this scope, so it is spelled
    # out: python decides a name is local for the whole function.
    uniform = np.full((16, 16), 33, dtype=np.uint8)
    for tag, clip, tiles in (("flat_clip_3_2x2", 3.0, (2, 2)),
                             ("flat_clip_40_8x8", 40.0, (8, 8))):
        engine = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
        add_windowed("clahe_" + tag, uniform, engine.apply(uniform), 0)

    # A size that does not divide by the tile grid, which is what a real
    # frame is. OpenCV pads when *either* dimension fails to divide and then
    # pads *both* by `tiles - (extent % tiles)`, so a dimension that already
    # divides gains a whole extra tile; `histogram.h` reproduces that rather
    # than rounding up, and this is the case that says so.
    odd = np.full((13, 17), 33, dtype=np.uint8)
    engine = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(4, 4))
    add_windowed("clahe_flat_odd_size", odd, engine.apply(odd), 0)

    odd_window = crop(gray)[:23, :31]
    engine = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    add_windowed("clahe_odd_size", odd_window, engine.apply(odd_window), 2)

    # ------------------------------------------------------------------
    # contours.h
    #
    # These produce numbers rather than images, so they are recorded as
    # their own section. `shapes` is a mask with an L, a ring, a diagonal
    # bar and a single pixel: shapes whose bounding box, area, hull and
    # minimum-area rectangle are all different from each other, and one
    # that eight-connectivity joins and four-connectivity does not.
    # ------------------------------------------------------------------
    shapes = np.zeros((24, 32), dtype=np.uint8)
    shapes[3:12, 3:6] = 255                    # the upright of an L
    shapes[9:12, 3:14] = 255                   # its foot
    shapes[3:12, 20:29] = 255                  # a filled square...
    shapes[5:10, 22:27] = 0                    # ...hollowed into a ring
    for step in range(10):                     # a diagonal bar
        shapes[15 + step // 2, 4 + step] = 255
    shapes[20, 28] = 255                       # a single pixel
    # two squares touching only at a corner: eight joins them, four does not
    shapes[16:19, 20:23] = 255
    shapes[19:22, 23:26] = 255

    shape_cases = []

    for how, flag in (("four", 4), ("eight", 8)):
        count, labels = cv2.connectedComponents(shapes, connectivity=flag)
        shape_cases.append({
            "name": "components_" + how,
            "components": int(count - 1),
            "labels": [int(v) for v in labels.reshape(-1)],
        })

    contours, _ = cv2.findContours(shapes, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    measured = []
    for c in contours:
        pts = c.reshape(-1, 2)          # (x, y) == (i, j)
        x, y, w, h = cv2.boundingRect(c)
        rr = cv2.minAreaRect(c)
        measured.append({
            "points": [int(v) for v in pts.reshape(-1)],
            "area": float(cv2.contourArea(c)),
            "bounds": [int(x), int(y), int(x + w), int(y + h)],
            "hull": [int(v) for v in
                     cv2.convexHull(c).reshape(-1)],
            "min_rect_area": float(rr[1][0] * rr[1][1]),
        })

    shape_cases.append({
        "name": "contours",
        "count": len(measured),
        "contours": measured,
    })

    # ------------------------------------------------------------------
    # match.h and layout.h
    #
    # Template matching is recorded as a float surface, scaled and offset
    # into the unsigned recording: the scores run -1 to 1 and the recording
    # carries integers, so a score of s becomes round((s + 1) * 10000).
    # The tolerance is then in ten-thousandths.
    # ------------------------------------------------------------------

    def as_scores(surface):
        return np.round((surface.astype(np.float64) + 1.0) * 10000.0
                        ).astype(np.int32)

    patch = window[6:14, 9:19]      # a piece of the window, so it matches
    surface = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
    add_windowed("match_ncc", window, as_scores(surface), 3)
    cases[-1]["pattern_left"] = 9
    cases[-1]["pattern_top"] = 6
    cases[-1]["pattern_width"] = 10
    cases[-1]["pattern_height"] = 8

    # A colour version, where OpenCV uses every channel together
    patch_rgb = window_rgb[6:14, 9:19]
    surface_rgb = cv2.matchTemplate(window_rgb, patch_rgb,
                                    cv2.TM_CCOEFF_NORMED)
    add_windowed("match_ncc_rgb", window_rgb, as_scores(surface_rgb), 3)
    cases[-1]["pattern_left"] = 9
    cases[-1]["pattern_top"] = 6
    cases[-1]["pattern_width"] = 10
    cases[-1]["pattern_height"] = 8

    add_windowed("hconcat", window,
                 cv2.hconcat([window, window[:, ::-1]]), 0)
    add_windowed("vconcat", window,
                 cv2.vconcat([window, window[::-1, :]]), 0)

    # ------------------------------------------------------------------
    # draw.h
    #
    # The geometry, which matches: rectangles, lines, circles and filled
    # polygons. Not text -- `font_5x7.h` is a bitmap font and OpenCV's is
    # Hershey's, so the glyph shapes differ by construction and there is
    # nothing to compare. LINE_8 because that is what every VIAME caller
    # asks for and what OpenCV defaults to.
    # ------------------------------------------------------------------

    def blank():
        return np.zeros((24, 32), dtype=np.uint8)

    canvas = blank()
    cv2.rectangle(canvas, (4, 3), (20, 15), 200, 1, cv2.LINE_8)
    add_windowed("draw_rect", blank(), canvas, 0)

    canvas = blank()
    cv2.rectangle(canvas, (4, 3), (20, 15), 200, -1, cv2.LINE_8)
    add_windowed("draw_rect_filled", blank(), canvas, 0)

    canvas = blank()
    for (a, b) in (((1, 1), (30, 22)), ((30, 2), (2, 20)),
                   ((0, 12), (31, 12)), ((16, 0), (16, 23))):
        cv2.line(canvas, a, b, 180, 1, cv2.LINE_8)
    add_windowed("draw_lines", blank(), canvas, 0)

    canvas = blank()
    cv2.circle(canvas, (16, 12), 9, 220, 1, cv2.LINE_8)
    add_windowed("draw_circle", blank(), canvas, 0)

    canvas = blank()
    cv2.circle(canvas, (16, 12), 9, 220, -1, cv2.LINE_8)
    add_windowed("draw_circle_filled", blank(), canvas, 0)

    canvas = blank()
    poly = np.array([[(5, 3), (28, 8), (20, 21), (8, 17)]], dtype=np.int32)
    cv2.fillPoly(canvas, poly, 150, cv2.LINE_8)
    add_windowed("draw_polygon", blank(), canvas, 0)

    return {
        "shapes_width": int(shapes.shape[1]),
        "shapes_height": int(shapes.shape[0]),
        "shapes_data": [int(v) for v in shapes.reshape(-1)],
        "shapes": shape_cases,
        "recorded": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "opencv": cv2.__version__,
        "note": "Recorded from OpenCV by tests/golden/image_ops/"
                "record_from_opencv.py. Tolerances are per case, in counts.",
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing recording")
    args = parser.parse_args()

    target = os.path.join(HERE, "opencv.json")

    if os.path.exists(target) and not args.force:
        print("{} already recorded; pass --force to re-record".format(target))
        return 1

    payload = record()

    with open(target, "w") as handle:
        json.dump(payload, handle, sort_keys=True, separators=( ",", ":" ))
        handle.write("\n")

    print("recorded {} cases into {}".format(len(payload["cases"]), target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
