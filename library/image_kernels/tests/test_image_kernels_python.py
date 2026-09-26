"""Python image kernels preserve OpenCV conventions for converted callers.

C++ callers which originally used VXL retain their separate resampling grid.
The Python resize binding uses OpenCV pixel centres, since detector and
classifier preprocessing were trained against that convention.
"""
import numpy as np
import pytest

from viame.image_kernels import (add_weighted, approx_poly, arc_length,
                                 bounding_rect,
                                 box_blur, clahe, contour_area, convex_hull,
                                 crop, distance_transform,
                                 intersect_convex, moments,
                                 demosaic, dilate, draw_circle, draw_line,
                                 find_contours, good_features_to_track,
                                 label_components, lucas_kanade,
                                 min_eigen_value, Mog2Background,
                                 corner_subpix, make_border,
                                 match_template, morphology,
                                 watershed,
                                 min_area_rect,
                                 draw_rect, draw_text, equalize, erode,
                                 fill_ellipse, fill_polygon,
                                 from_hls, from_hsv, from_lab, gaussian_blur,
                                 normalize, optical_flow, remap, resize,
                                 resize_area,
                                 swap_channels, text_size, to_gray, to_hls,
                                 to_hsv, to_lab, to_rgb, warp_affine,
                                 warp_perspective, dilate)


def _frame(width=64, height=48):
    y, x = np.mgrid[0:height, 0:width]
    return np.stack([(x * 4) % 256, (y * 5) % 256, ((x + y) * 3) % 256],
                    axis=-1).astype(np.uint8)


def test_resize_gives_the_requested_size():
    out = resize(_frame(), 32, 24)
    assert out.shape == (24, 32, 3)


def test_resize_of_a_flat_image_is_flat():
    flat = np.full((20, 20, 3), 77, dtype=np.uint8)
    out = resize(flat, 10, 10)
    assert out.shape == (10, 10, 3)
    assert out.min() == 77 and out.max() == 77


def test_crop_matches_a_numpy_slice():
    image = _frame()
    assert np.array_equal(crop(image, 10, 5, 20, 15), image[5:20, 10:30])


def test_crop_clamps_to_the_image():
    image = _frame(20, 20)
    out = crop(image, 15, 15, 100, 100)
    assert out.shape[0] <= 5 and out.shape[1] <= 5


def test_to_gray_uses_the_expected_weights():
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)
    image[0, 1] = (0, 255, 0)
    image[0, 2] = (0, 0, 255)
    assert list(to_gray(image)[0]) == [76, 150, 29]


def test_to_gray_needs_three_channels():
    with pytest.raises(Exception):
        to_gray(np.zeros((4, 4), dtype=np.uint8))


def test_swap_channels_is_its_own_inverse():
    image = _frame()
    assert np.array_equal(swap_channels(swap_channels(image)), image)
    assert np.array_equal(swap_channels(image)[..., 0], image[..., 2])


def test_to_rgb_repeats_the_single_channel():
    gray = to_gray(_frame())
    rgb = to_rgb(gray)
    assert rgb.shape == gray.shape + (3,)
    assert np.array_equal(rgb[..., 0], rgb[..., 1])


def test_a_two_dimensional_image_stays_two_dimensional():
    gray = to_gray(_frame())
    assert resize(gray, 16, 12).shape == (12, 16)


# ---------------------------------------------------------------------------
# Colour spaces
#
# The scaling follows the **type**, exactly as `cv::cvtColor` makes it: an
# 8-bit image has hue halved into 0..179 with saturation and value over
# 0..255, because a byte cannot hold degrees, and a float image keeps hue in
# 0..360 with the other two over 0..1.
#
# Both forms were always in `color.h`. Only the 8-bit one was bound, which is
# why the netharn augmenters stayed on cv2: they work in float, and handing
# their 0..360 hue to the 8-bit form silently halves it.

@pytest.mark.parametrize("forward,inverse", [(to_hsv, from_hsv),
                                             (to_hls, from_hls),
                                             (to_lab, from_lab)])
def test_a_colour_space_round_trips(forward, inverse):
    frame = _frame()
    back = inverse(forward(frame))
    assert back.shape == frame.shape
    # 8-bit L*a*b* is lossy enough that OpenCV loses as much; what is checked
    # is that nothing is grossly wrong, not that it is exact.
    assert np.abs(back.astype(int) - frame.astype(int)).max() <= 24


def test_lab_is_opencvs_fixed_point_conversion_and_not_the_formula():
    """`to_lab` reproduces `cv::cvtColor`'s integer path, not the real-valued
    definition, and the two disagree by up to two counts. Checked here at
    the values that separate them.

    `rgb -> lab` was compared against cv2 5.0.0 over all 16777216 8-bit
    triples when this landed: identical on every one. What the table below
    holds is the part a formula cannot get right -- the first three rows turn
    on a single cube-root table entry whose product lands a ten-thousandth
    above a rounding tie, where OpenCV rounds down and arithmetic rounds up.
    """
    cases = [([0, 6, 98], [24, 163, 77]),
             ([0, 22, 138], [45, 169, 64]),
             ([32, 1, 86], [24, 163, 85]),
             ([0, 0, 0], [0, 128, 128]),
             ([255, 255, 255], [255, 128, 128]),
             ([255, 0, 0], [136, 208, 195]),
             ([0, 255, 0], [224, 42, 211]),
             ([0, 0, 255], [82, 207, 20])]

    frame = np.array([[rgb for rgb, _ in cases]], dtype=np.uint8)
    expected = np.array([[lab for _, lab in cases]], dtype=np.uint8)
    assert to_lab(frame).tolist() == expected.tolist()


def test_from_lab_is_opencvs_integer_path_and_not_its_float_one():
    """`cv::cvtColor`'s 8-bit L*a*b*-to-RGB is separate integer code, not its
    float path rounded: cv2's own float answer, rounded to a byte, disagrees
    with its 8-bit answer by a count on 2.8% of triples. `from_lab` follows
    the integer one, and was compared against cv2 5.0.0 over all 16777216
    triples when it landed -- identical on every one.

    The values below are a sample of that, taken at the ends and at the
    primaries' own L*a*b*, where the three tables and the 14-bit fixed point
    all get exercised.
    """
    cases = [([0, 128, 128], [0, 0, 0]),
             ([255, 128, 128], [255, 255, 255]),
             ([136, 208, 195], [255, 2, 1]),
             ([224, 42, 211], [7, 255, 3]),
             ([82, 207, 20], [0, 1, 255]),
             ([24, 163, 77], [1, 7, 98]),
             ([45, 169, 64], [0, 23, 139])]

    frame = np.array([[lab for lab, _ in cases]], dtype=np.uint8)
    expected = np.array([[rgb for _, rgb in cases]], dtype=np.uint8)
    assert from_lab(frame).tolist() == expected.tolist()


@pytest.mark.parametrize("convert", [to_hsv, to_hls, to_lab])
def test_a_colour_space_keeps_the_shape(convert):
    assert convert(_frame(32, 16)).shape == (16, 32, 3)


@pytest.mark.parametrize("convert", [to_hsv, to_hls, to_lab, from_hsv,
                                     from_hls, from_lab])
def test_a_colour_space_needs_three_channels(convert):
    with pytest.raises(ValueError):
        convert(np.zeros((8, 8), dtype=np.uint8))


def test_hue_of_the_primaries_is_on_the_opencv_scale():
    """Red 0, green 60 and blue 120 -- degrees halved to fit a byte."""
    primaries = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]],
                         dtype=np.uint8)
    assert list(to_hsv(primaries)[0, :, 0]) == [0, 60, 120]


def test_grey_has_no_saturation():
    grey = np.full((4, 4, 3), 128, dtype=np.uint8)
    assert to_hsv(grey)[..., 1].max() == 0
    assert to_hls(grey)[..., 2].max() == 0


# ---------------------------------------------------------------------------
# Filtering, histograms and morphology
#
# Measured against cv2 when written, on a random 64 by 96 frame: gaussian_blur,
# box_blur, add_weighted, normalize, equalize, erode, dilate and demosaic are
# all **bit identical**. So is clahe, since P7-T04c -- it was a grey level out
# until the scaling and the interpolation moved to float and the rounding to
# half-to-even, which is what OpenCV does; 192 configurations agree exactly,
# over eight shapes including one that does not divide by its tile grid. Those
# are strong enough agreements to assert shape and invariants here and leave
# the pixel comparison to tests/golden.

def _gray(width=64, height=48):
    y, x = np.mgrid[0:height, 0:width]
    return (((x * 7) % 256) ^ ((y * 11) % 256)).astype(np.uint8)


@pytest.mark.parametrize("blur", [lambda a: gaussian_blur(a, 5),
                                  lambda a: box_blur(a, 5)])
def test_a_blur_keeps_the_shape_and_narrows_the_range(blur):
    frame = _gray()
    out = blur(frame)
    assert out.shape == frame.shape
    assert np.ptp(out) <= np.ptp(frame)


def test_a_blur_of_a_flat_image_is_flat():
    flat = np.full((20, 20), 90, dtype=np.uint8)
    assert gaussian_blur(flat, 5).min() == 90
    assert gaussian_blur(flat, 5).max() == 90


def test_add_weighted_averages():
    dark = np.full((8, 8), 40, dtype=np.uint8)
    light = np.full((8, 8), 200, dtype=np.uint8)
    assert add_weighted(dark, 0.5, light, 0.5)[0, 0] == 120


def test_add_weighted_wants_one_size():
    with pytest.raises(ValueError):
        add_weighted(np.zeros((4, 4), np.uint8), 1.0,
                     np.zeros((5, 5), np.uint8), 1.0)


def test_normalize_spans_the_range():
    out = normalize(_gray() // 4 + 30, 0, 255)
    assert out.min() == 0 and out.max() == 255


def test_clahe_is_opencvs_float_arithmetic_and_its_rounding():
    """Both halves of the agreement, on the smallest case that shows them.

    `clahe` was a grey level away from `cv2.createCLAHE` until two things
    changed together: the cumulative histogram is scaled in float and the
    bilinear blend is computed in float and grouped across-then-down, and the
    result is rounded half to even rather than half away from zero. Either one
    in double and tens of pixels in a frame come out a count off.

    A 1 by 1 grid takes the interpolation out, so this pins the scaling; the
    2 by 2 case exercises both.
    """
    frame = _gray(16, 12)
    assert clahe(frame, 3.0, 1, 1)[:2, :4].tolist() == [[4, 17, 29, 42],
                                                        [25, 28, 12, 60]]
    assert clahe(frame, 3.0, 2, 2)[:2, :4].tolist() == [[11, 37, 58, 85],
                                                        [48, 53, 27, 117]]


def test_equalize_and_clahe_keep_the_shape():
    frame = _gray()
    assert equalize(frame).shape == frame.shape
    assert clahe(frame, 2.0, 8, 8).shape == frame.shape


def test_erode_darkens_and_dilate_brightens():
    frame = _gray()
    assert erode(frame, "rect", 3, 3).mean() <= frame.mean()
    assert dilate(frame, "rect", 3, 3).mean() >= frame.mean()


def test_an_unknown_structuring_element_is_refused():
    with pytest.raises(ValueError):
        erode(_gray(), "hexagon", 3, 3)


def test_float_hsv_keeps_degrees():
    """float32 gets 0..360 and 0..1, where uint8 gets 0..179 and 0..255."""
    rgb = np.array([[[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [0.5, 0.5, 0.5]]], dtype=np.float32)

    hsv = to_hsv(rgb)

    assert hsv.dtype == np.float32
    assert hsv[0, 0, 0] == pytest.approx(0.0)      # red
    assert hsv[0, 1, 0] == pytest.approx(120.0)    # green
    assert hsv[0, 2, 0] == pytest.approx(240.0)    # blue
    assert hsv[0, 3, 1] == pytest.approx(0.0)      # grey has no saturation
    assert hsv[0, 3, 2] == pytest.approx(0.5)


def test_float_hsv_round_trips_far_better_than_eight_bit():
    """No 0..179 halving and no 0..255 quantisation to lose."""
    rng = np.random.default_rng(4)
    rgb = rng.random((12, 16, 3)).astype(np.float32)

    back = from_hsv(to_hsv(rgb))

    assert back.dtype == np.float32
    assert np.abs(back - rgb).max() < 1e-5


def test_the_two_hsv_scalings_agree_once_rescaled():
    """The same conversion, said in two units.

    Converting a frame as float and as bytes must give the same colour: hue
    halved, the other two over 255. Held loosely because the 8-bit form
    rounds each channel to an integer, which is a whole degree of hue.
    """
    frame = _frame()

    as_bytes = to_hsv(frame).astype(np.float64)
    as_float = to_hsv(np.ascontiguousarray(
        frame.astype(np.float32) / 255.0)).astype(np.float64)

    assert np.abs(as_bytes[:, :, 0] - as_float[:, :, 0] / 2.0).max() <= 1.0
    assert np.abs(as_bytes[:, :, 1] - as_float[:, :, 1] * 255.0).max() <= 1.0
    assert np.abs(as_bytes[:, :, 2] - as_float[:, :, 2] * 255.0).max() <= 1.0


def test_an_unknown_border_is_refused():
    """`wrap` used to be the example here, and is now a border rule.

    Which is the point of naming a mode that is genuinely absent instead:
    numpy's pad offers `linear_ramp`, `mean` and several more that OpenCV has
    no equivalent for, and asking for one must fail rather than quietly
    padding some other way.
    """
    with pytest.raises(ValueError):
        gaussian_blur(_gray(), 5, 0.0, "linear_ramp")


def test_wrap_is_a_border_rule_everywhere_it_is_offered():
    """Added with `make_border`, and `as_border` serves every caller."""
    assert gaussian_blur(_gray(), 5, 0.0, "wrap").shape == _gray().shape


def test_demosaic_gives_three_planes():
    mosaic = _gray(32, 24)
    assert demosaic(mosaic, "BG").shape == (24, 32, 3)


def test_demosaic_names_the_mosaic_not_opencvs_spelling():
    """`BG` is blue at (0, 0), which is cv2.COLOR_BayerRG2RGB, not BayerBG2RGB.

    The two read the letters in opposite orders; getting this backwards
    swaps red and blue on every debayered frame, which is the kind of bug
    that survives review because the image still looks like an image.
    """
    mosaic = np.zeros((4, 4), dtype=np.uint8)
    mosaic[0::2, 0::2] = 255          # blue sites of a BG mosaic
    out = demosaic(mosaic, "BG")
    assert out[0, 0, 2] > out[0, 0, 0]


def test_an_unknown_bayer_pattern_is_refused():
    with pytest.raises(ValueError):
        demosaic(_gray(8, 8), "XY")


# ---------------------------------------------------------------------------
# Drawing
#
# These write into the array they are given, as cv2.fillPoly and friends do.

def test_fill_ellipse_covers_the_exact_area():
    """The mathematical ellipse, not OpenCV's polygonal fill of it.

    `cv::ellipse` fills the polygon it approximates the ellipse with, and
    that runs a boundary ring fatter -- 4 to 10 per cent more pixels over the
    shapes measured, a circle of radius 20 coming out 1307 there against
    1257 here. 1257 is the right answer: the exact area is 1256.6.
    """
    image = np.zeros((80, 80), dtype=np.uint8)

    fill_ellipse(image, 40, 40, 20, 20, 1)

    assert int(image.sum()) == pytest.approx(np.pi * 400, abs=4)


def test_fill_ellipse_respects_its_two_axes():
    image = np.zeros((60, 100), dtype=np.uint8)

    fill_ellipse(image, 50, 30, 30, 12, 1)

    rows = np.where(image.any(axis=1))[0]
    columns = np.where(image.any(axis=0))[0]

    assert columns.min() == 20 and columns.max() == 80
    assert rows.min() == 18 and rows.max() == 42


def test_fill_ellipse_turns():
    """A quarter turn swaps the axes, which is the cheapest check that the
    angle is applied at all and applied about the centre."""
    upright = np.zeros((100, 100), dtype=np.uint8)
    fill_ellipse(upright, 50, 50, 30, 12, 1)

    turned = np.zeros((100, 100), dtype=np.uint8)
    fill_ellipse(turned, 50, 50, 30, 12, 1, 90.0)

    assert np.array_equal(turned, upright.T)


def test_fill_ellipse_clips_at_the_edge():
    image = np.zeros((60, 60), dtype=np.uint8)

    fill_ellipse(image, 5, 5, 20, 20, 1)

    assert image[0, 0] == 1
    assert image.sum() < np.pi * 400


def test_a_degenerate_ellipse_is_a_line():
    """Zero on one axis draws the line through the centre, as OpenCV does,
    rather than drawing nothing."""
    image = np.zeros((20, 40), dtype=np.uint8)

    fill_ellipse(image, 20, 10, 8, 0, 1)

    assert image[10].sum() == 17
    assert image.sum() == 17


def test_fill_polygon_writes_in_place():
    canvas = np.zeros((20, 30), dtype=np.uint8)
    fill_polygon(canvas, np.array([[2.0, 2.0], [20.0, 3.0], [15.0, 15.0]]), 255)
    assert canvas.max() == 255


def test_fill_polygon_ignores_a_degenerate_polygon():
    canvas = np.zeros((20, 30), dtype=np.uint8)
    fill_polygon(canvas, np.array([[2.0, 2.0], [20.0, 3.0]]), 255)
    assert canvas.max() == 0


def test_points_must_be_pairs():
    canvas = np.zeros((20, 30), dtype=np.uint8)
    with pytest.raises(ValueError):
        fill_polygon(canvas, np.zeros((4, 3)), 255)


def test_draw_rect_bounds_are_exclusive():
    """cv2.rectangle's second corner is inclusive; this one is not.

    The pixels are otherwise identical, so this off by one is the whole
    difference between the two and is worth a test of its own.
    """
    canvas = np.zeros((20, 20), dtype=np.uint8)
    draw_rect(canvas, 2, 2, 10, 10, 255, 1)
    assert canvas[2, 9] == 255
    assert canvas[2, 10] == 0


def test_draw_rect_fills_at_a_negative_thickness():
    canvas = np.zeros((20, 20), dtype=np.uint8)
    draw_rect(canvas, 2, 2, 10, 10, 255, -1)
    assert canvas[5, 5] == 255


def test_draw_circle_and_line_mark_the_canvas():
    canvas = np.zeros((40, 40), dtype=np.uint8)
    draw_circle(canvas, 20, 20, 8, 255, 1)
    assert canvas[20, 12] == 255

    line = np.zeros((40, 40), dtype=np.uint8)
    draw_line(line, 0, 0, 39, 39, 255)
    assert line[20, 20] == 255


def test_draw_text_places_by_the_top_left():
    canvas = np.zeros((20, 80), dtype=np.uint8)
    draw_text(canvas, "VIAME", 2, 2, 255)
    width, height = text_size("VIAME")
    assert canvas[:2].max() == 0             # nothing above the top
    assert canvas[2:2 + height, 2:2 + width].max() == 255


def test_text_size_grows_with_the_scale():
    assert text_size("hi", 2)[0] > text_size("hi", 1)[0]


def test_colour_may_be_a_scalar_or_per_plane():
    canvas = np.zeros((10, 10, 3), dtype=np.uint8)
    draw_rect(canvas, 1, 1, 8, 8, 200, -1)
    assert list(canvas[4, 4]) == [200, 200, 200]

    draw_rect(canvas, 1, 1, 8, 8, [10, 20, 30], -1)
    assert list(canvas[4, 4]) == [10, 20, 30]


# ---------------------------------------------------------------------------
# Warping
#
# Against cv2 on a smooth frame, ignoring the border where the border rule
# dominates: remap and the two warps agree to **one grey level** on bilinear
# and on bicubic. Nearest is the exception -- it differs on about 7% of
# pixels, all of them ties, because `std::lround` rounds a half away from
# zero and OpenCV's `cvRound` rounds it to even.

def _smooth(width=64, height=48):
    y, x = np.mgrid[0:height, 0:width]
    return np.stack([np.sin(x / 9.0) * 110 + 128,
                     np.cos(y / 7.0) * 110 + 128,
                     (x + y) * 2 % 256], axis=-1).astype(np.uint8)


def _identity_maps(width=64, height=48):
    mx, my = np.meshgrid(np.arange(width, dtype=np.float32),
                         np.arange(height, dtype=np.float32))
    return mx, my


def test_remap_through_the_identity_returns_the_image():
    frame = _smooth()
    mx, my = _identity_maps()
    assert np.array_equal(remap(frame, mx, my), frame)


def test_remap_takes_the_maps_size():
    frame = _smooth()
    mx, my = _identity_maps(20, 10)
    assert remap(frame, mx, my).shape == (10, 20, 3)


def test_remap_wants_two_maps_of_one_size():
    frame = _smooth()
    mx, _ = _identity_maps(20, 10)
    _, my = _identity_maps(30, 10)
    with pytest.raises(ValueError):
        remap(frame, mx, my)


@pytest.mark.parametrize("how", ["nearest", "bilinear", "bicubic", "area"])
def test_every_interpolation_is_accepted(how):
    frame = _smooth()
    mx, my = _identity_maps()
    assert remap(frame, mx, my, how).shape == frame.shape


def test_an_unknown_interpolation_is_refused():
    frame = _smooth()
    mx, my = _identity_maps()
    with pytest.raises(ValueError):
        remap(frame, mx, my, "lanczos")


def test_bicubic_is_not_bilinear():
    """Otherwise the cubic call sites would be silently downgraded.

    Fifteen `cv2.remap` and `cv2.resize` call sites ask for INTER_CUBIC; if
    `"bicubic"` quietly fell through to the bilinear kernel this test is the
    only thing that would notice.
    """
    frame = _smooth()
    mx, my = _identity_maps()
    shifted_x, shifted_y = mx + 0.37, my + 0.21
    linear = remap(frame, shifted_x, shifted_y, "bilinear")
    cubic = remap(frame, shifted_x, shifted_y, "bicubic")
    assert not np.array_equal(linear, cubic)


def test_warp_perspective_through_the_identity_returns_the_image():
    frame = _smooth()
    assert np.array_equal(warp_perspective(frame, np.eye(3)), frame)


def test_warp_affine_through_the_identity_returns_the_image():
    frame = _smooth()
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.array_equal(warp_affine(frame, identity), frame)


def test_warp_affine_wants_a_two_by_three():
    with pytest.raises(ValueError):
        warp_affine(_smooth(), np.eye(3))


def test_warp_perspective_wants_a_three_by_three():
    with pytest.raises(ValueError):
        warp_perspective(_smooth(), np.zeros((2, 3)))


def test_a_warp_takes_the_requested_size():
    assert warp_perspective(_smooth(), np.eye(3), 20, 10).shape == (10, 20, 3)


# ---------------------------------------------------------------------------
# Pixel types
#
# `forcecast` is deliberately not on these bindings. With it, a float32 depth
# map handed to a uint8 binding is silently truncated -- and a silently wrong
# depth map is worse than a TypeError. So the types a VIAME pipeline actually
# carries are bound explicitly: uint8, uint16 for the 16-bit pipelines
# (`common_default_input_16bit.pipe`), and float32 for a depth or disparity
# stage. Anything else raises.

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_the_resamplers_carry_the_pixel_type(dtype):
    frame = (_smooth() / 2).astype(dtype)
    assert resize(frame, 32, 24).dtype == dtype
    assert resize_area(frame, 32, 24).dtype == dtype
    assert crop(frame, 0, 0, 20, 10).dtype == dtype

    mx, my = _identity_maps()
    assert remap(frame, mx, my).dtype == dtype
    assert warp_perspective(frame, np.eye(3)).dtype == dtype


def test_a_float32_map_keeps_its_range():
    """The point of the float overload: 5000.0 must not come back as 255."""
    depth = np.full((8, 8), 5000.0, dtype=np.float32)
    assert resize(depth, 4, 4).max() == pytest.approx(5000.0)


def test_a_sixteen_bit_frame_keeps_its_range():
    frame = np.full((8, 8), 40000, dtype=np.uint16)
    assert resize(frame, 4, 4).max() == 40000


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_drawing_takes_the_frame_types(dtype):
    canvas = np.zeros((20, 20), dtype=dtype)
    draw_rect(canvas, 2, 2, 10, 10, 300 if dtype is np.uint16 else 200, -1)
    assert canvas[5, 5] > 0

    fill_polygon(canvas, np.array([[1.0, 1.0], [15.0, 2.0], [10.0, 14.0]]), 7)
    assert canvas.max() > 0


def test_an_unsupported_pixel_type_is_refused_not_cast():
    with pytest.raises(TypeError):
        resize(np.zeros((8, 8), np.float64), 4, 4)
    with pytest.raises(TypeError):
        resize(np.zeros((8, 8), np.int32), 4, 4)


# ---------------------------------------------------------------------------
# Contours
#
# Against cv2 on a mask of a rectangle and a disc: the traced **point sets**
# are identical, as are the areas, the bounding boxes, the convex hulls, the
# minimum-area rectangles and the component partitions. The one difference is
# length -- each contour here is closed, repeating its first point, where
# cv2's is open.

def _two_shapes():
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[10:30, 12:40] = 1
    ys, xs = np.mgrid[0:60, 0:80]
    mask[((xs - 60) ** 2 + (ys - 42) ** 2) <= 81] = 1
    return mask


def test_find_contours_traces_each_shape():
    contours = find_contours(_two_shapes())
    assert len(contours) == 2
    assert all(c.shape[1] == 2 for c in contours)


def test_a_contour_is_closed():
    """cv2 leaves its contours open; these repeat the first point."""
    contour = max(find_contours(_two_shapes()), key=len)
    assert np.array_equal(contour[0], contour[-1])


def test_contour_area_and_bounding_rect_match_the_shape():
    contour = max(find_contours(_two_shapes()), key=len)
    assert contour_area(contour) == pytest.approx(513.0)
    assert bounding_rect(contour) == (12, 10, 28, 20)


def test_an_empty_mask_traces_nothing():
    assert find_contours(np.zeros((10, 10), dtype=np.uint8)) == []


def test_convex_hull_drops_the_interior_point():
    points = np.array([[5.0, 5.0], [40.0, 7.0], [38.0, 30.0], [7.0, 28.0],
                       [20.0, 15.0]])
    assert len(convex_hull(points)) == 4


def test_min_area_rect_of_an_axis_aligned_square():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    box = min_area_rect(square)
    assert box["size"][0] * box["size"][1] == pytest.approx(100.0, rel=0.02)


def test_approx_poly_reduces_a_rectangle_to_four_corners():
    contour = max(find_contours(_two_shapes()), key=len)
    assert len(approx_poly(contour, 3.0)) == 4


def test_label_components_counts_the_background():
    count, labels = label_components(_two_shapes(), 8)
    assert count == 3                      # two shapes plus the background
    assert labels.shape == (60, 80)
    assert set(np.unique(labels)) == {0, 1, 2}


def test_arc_length_of_a_rectangle():
    """The polygon through the pixel centres, as `contour_area` measures too.

    The traced rectangle spans 12..39 and 10..29, so it is 27 by 19 at its
    centres and goes twice round: 92, not the 96 a pixel count suggests.
    """
    contour = max(find_contours(_two_shapes()), key=len)
    assert arc_length(contour, True) == pytest.approx(92.0)


def test_arc_length_open_drops_the_closing_edge():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert arc_length(square, True) == pytest.approx(40.0)
    assert arc_length(square, False) == pytest.approx(30.0)


def test_moments_give_the_centroid_of_a_square():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    m = moments(square)

    assert m["m00"] == pytest.approx(100.0)
    assert m["m10"] / m["m00"] == pytest.approx(5.0)
    assert m["m01"] / m["m00"] == pytest.approx(5.0)


def test_moments_find_an_offset_centroid():
    """An L is not centred on its bounding box, which is the point of asking."""
    shape = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 10.0],
                      [10.0, 10.0], [10.0, 30.0], [0.0, 30.0]])
    m = moments(shape)

    assert m["m00"] == pytest.approx(500.0)
    assert m["m10"] / m["m00"] == pytest.approx(11.0)
    assert m["m01"] / m["m00"] == pytest.approx(11.0)


def test_intersect_convex_overlapping_squares():
    a = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    b = np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]])

    area, points = intersect_convex(a, b)

    assert area == pytest.approx(25.0)
    assert len(points) == 4


def test_intersect_convex_keeps_sub_pixel_crossings():
    """The vertices of an overlap are where two edges cross.

    Rounding them to the pixel grid is a whole pixel of error on each, which
    is why this works in doubles rather than through `contour_points`.
    """
    a = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    b = np.array([[2.5, 2.5], [12.5, 2.5], [12.5, 12.5], [2.5, 12.5]])

    area, points = intersect_convex(a, b)

    assert area == pytest.approx(56.25)
    assert np.any(np.abs(points - np.round(points)) > 0.1)


def test_intersect_convex_is_the_same_either_way_round():
    """Winding is normalised, so a clockwise argument is not silently empty."""
    a = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    clockwise = a[::-1].copy()
    b = np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]])

    assert intersect_convex(clockwise, b)[0] == pytest.approx(25.0)
    assert intersect_convex(b, a)[0] == pytest.approx(25.0)


def test_intersect_convex_of_disjoint_squares_is_empty():
    a = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    b = np.array([[20.0, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 30.0]])

    area, points = intersect_convex(a, b)

    assert area == 0.0
    assert len(points) == 0


def test_intersect_convex_contained_square_is_the_smaller_one():
    outer = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])
    inner = np.array([[5.0, 5.0], [10.0, 5.0], [10.0, 10.0], [5.0, 10.0]])

    assert intersect_convex(outer, inner)[0] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# The float32 overloads
#
# Seven kernels were bound for uint8 alone although the C++ behind each is a
# template: a caller with a float image -- a depth map, a backscatter
# estimate, anything mid-computation -- had to go back to cv2 for them.
#
# Against cv2 on a random float32 frame: erode, dilate and add_weighted are
# exact, gaussian_blur is within 3.1e-5, box_blur 1.6e-5 and normalize 6.0e-8,
# every one of those being float32's own rounding rather than a difference in
# what is computed.
#
# `clahe` is the exception and is bound for uint8 and uint16 only. It
# static_asserts on an integer pixel and is right to: it equalises a
# histogram, which needs a bounded range of discrete levels to build one
# over, and `cv2.createCLAHE` takes 8 and 16 bit for the same reason.

# ---------------------------------------------------------------------------
# Distance transform
#
# Against cv2 with DIST_L2 and a mask of 3: bit identical on a speckle, an
# all-foreground mask, an all-background one and a shape running off the
# edge, and within 4.8e-7 -- float32's own rounding -- on a rectangle, a disc
# and an annulus.
#
# The mode is a **chamfer approximation** and not the Euclidean distance the
# name suggests: two passes with Borgefors' step costs, 0.955 sideways and
# 1.3693 diagonally, which are not 1 and sqrt(2). They are fitted to minimise
# the worst error rather than to be exact along an axis, so even a horizontal
# run comes out 4.5% short -- which the first test below states outright,
# because it is the thing a reader will assume is a bug.

def test_the_distance_is_a_chamfer_and_not_euclidean():
    """A 1-pixel step costs 0.955, not 1. That is the mode, not an error.

    One zero column and the rest foreground, so the nearest zero really is
    along the row. A one-row strip would not do: every pixel of it is next to
    the background above and below, so the whole strip is 0.955 and the step
    cost never shows.
    """
    mask = np.ones((6, 8), dtype=np.uint8)
    mask[:, 0] = 0

    out = distance_transform(np.ascontiguousarray(mask))

    assert out.dtype == np.float32
    assert list(out[3][:4]) == pytest.approx([0.0, 0.955, 1.910, 2.865],
                                             abs=1e-5)


def test_background_is_zero_and_shapes_grow_inward():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1

    out = distance_transform(np.ascontiguousarray(mask))

    assert out[0, 0] == 0.0
    assert out[4, 10] == 0.0                       # just outside the square
    # The deepest point is the middle, five pixels in
    assert out[9, 9] == pytest.approx(out.max())
    assert out.max() == pytest.approx(5 * 0.955, abs=1e-4)


def test_a_mask_with_no_background_saturates():
    """There is no zero to be distant from, so there is no answer.

    OpenCV returns a saturated float here and so does this; what matters is
    that it is obviously not a distance rather than quietly a small one.
    """
    out = distance_transform(np.ones((8, 8), dtype=np.uint8))

    assert np.all(out > 1e30)


def test_the_image_edge_is_not_background():
    """A shape running off the edge is not made shallow by the edge.

    The pixels beyond are unknown, not empty, which is OpenCV's rule and the
    right one -- the other way round, every mask touching a border would read
    as thin there.
    """
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, 5:] = 1

    out = distance_transform(np.ascontiguousarray(mask))

    # Depth grows away from the boundary at column 5 and is not reset by the
    # right-hand edge of the image.
    assert out[5, 9] > out[5, 6]


def test_distance_transform_wants_one_plane():
    with pytest.raises((ValueError, TypeError)):
        distance_transform(np.zeros((8, 8, 3), dtype=np.uint8))


def _float_frame():
    rng = np.random.default_rng(21)
    return np.ascontiguousarray((rng.random((24, 32)) * 255).astype(np.float32))


@pytest.mark.parametrize("call", [
    lambda f: erode(f, "rect", 5, 5),
    lambda f: dilate(f, "rect", 5, 5),
    lambda f: gaussian_blur(f, 7, 2.0),
    lambda f: box_blur(f, 5),
    lambda f: normalize(f, 0.0, 1.0),
])
def test_a_float_image_stays_float(call):
    out = call(_float_frame())
    assert out.dtype == np.float32
    assert out.shape == (24, 32)


def test_float_erosion_keeps_the_minimum_of_the_window():
    """Which is the whole of what an erosion is, and is checkable exactly."""
    frame = np.zeros((5, 5), dtype=np.float32)
    frame[2, 2] = -7.5
    frame[0, 0] = 3.25

    out = erode(np.ascontiguousarray(frame), "rect", 3, 3)

    # The low value spreads over its three by three neighbourhood
    assert out[1, 1] == pytest.approx(-7.5)
    assert out[3, 3] == pytest.approx(-7.5)
    # and a float erosion does not clamp at zero the way a uint8 one must
    assert out.min() == pytest.approx(-7.5)


def test_float_normalize_reaches_both_ends():
    frame = _float_frame()

    out = normalize(frame, 0.0, 1.0)

    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_float_add_weighted_does_not_clamp():
    """A uint8 sum saturates at 255; a float one has nothing to saturate to."""
    a = np.full((4, 4), 200.0, dtype=np.float32)
    b = np.full((4, 4), 200.0, dtype=np.float32)

    out = add_weighted(np.ascontiguousarray(a), 1.0,
                       np.ascontiguousarray(b), 1.0, 0.0)

    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(400.0)


def test_clahe_takes_sixteen_bit():
    rng = np.random.default_rng(5)
    frame = np.ascontiguousarray(
        (rng.random((32, 48)) * 65535).astype(np.uint16))

    out = clahe(frame, 2.0, 8, 8)

    assert out.dtype == np.uint16
    assert out.shape == frame.shape


def test_clahe_refuses_a_float_image():
    """Not an oversight: a histogram needs discrete levels to bin into."""
    with pytest.raises(TypeError):
        clahe(_float_frame(), 2.0, 8, 8)


def test_a_contour_must_be_pairs():
    with pytest.raises(ValueError):
        contour_area(np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# Template matching, morphology compositions and borders
#
# All three are exact against cv2 on a random frame, which is what makes the
# `iterations` test below worth having: the first attempt repeated the
# erode/dilate **pair** n times, which matches cv2 at one iteration and is
# over a hundred grey levels out at three. Anything that only checked
# `iterations=1` would have shipped it.

def test_match_template_finds_the_patch_it_was_cut_from():
    frame = _gray(64, 48)
    patch = np.ascontiguousarray(frame[10:22, 15:31])
    scores = match_template(frame, patch)
    assert scores.shape == (48 - 12 + 1, 64 - 16 + 1)
    assert np.unravel_index(scores.argmax(), scores.shape) == (10, 15)
    assert scores.max() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("operation", ["open", "close"])
@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_morphology_keeps_the_shape(operation, iterations):
    frame = _gray()
    out = morphology(frame, operation, "rect", 3, 3, iterations)
    assert out.shape == frame.shape


def test_opening_darkens_and_closing_brightens():
    frame = _gray()
    assert morphology(frame, "open").mean() <= frame.mean()
    assert morphology(frame, "close").mean() >= frame.mean()


def test_an_unknown_morphology_is_refused():
    with pytest.raises(ValueError):
        morphology(_gray(), "gradient")


def test_make_border_grows_by_the_margins():
    frame = _gray(20, 16)
    out = make_border(frame, 5, 7, 3, 9, 0)
    assert out.shape == (16 + 5 + 7, 20 + 3 + 9)
    assert np.array_equal(out[5:5 + 16, 3:3 + 20], frame)
    assert out[0, 0] == 0


def test_make_border_takes_a_value_per_plane():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    out = make_border(frame, 2, 2, 2, 2, [7, 8, 9])
    assert list(out[0, 0]) == [7, 8, 9]


# The padding modes, on a single row so the rule is readable. The source is
# `a b c d e` as 1..5, padded three each side:
#
#   replicate     a a a | a b c d e | e e e
#   reflect       c b a | a b c d e | e d c
#   reflect_101   d c b | a b c d e | d c b
#   wrap          c d e | a b c d e | a b c
#
# `reflect` and `reflect_101` differ by whether the edge pixel is repeated,
# which is the distinction numpy calls `symmetric` and `reflect` -- the same
# two words for the other two rules. Getting them the wrong way round shifts
# a padded image by one pixel and nothing else, which is why they are spelled
# out here rather than trusted.
@pytest.mark.parametrize("mode,expected", [
    ("replicate",   [1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5]),
    ("reflect",     [3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3]),
    ("reflect_101", [4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2]),
    ("wrap",        [3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]),
])
def test_make_border_pads_by_the_rule(mode, expected):
    row = np.array([[1, 2, 3, 4, 5]], dtype=np.uint8)
    out = make_border(row, 0, 0, 3, 3, 0, mode)
    assert list(out[0]) == expected


def test_make_border_constant_ignores_the_far_side():
    row = np.array([[1, 2, 3, 4, 5]], dtype=np.uint8)
    out = make_border(row, 0, 0, 2, 2, 9, "constant")
    assert list(out[0]) == [9, 9, 1, 2, 3, 4, 5, 9, 9]


def test_make_border_defaults_to_constant():
    """Every caller written before the modes existed passed no mode."""
    row = np.array([[1, 2, 3, 4, 5]], dtype=np.uint8)
    assert np.array_equal(make_border(row, 0, 0, 2, 2, 9),
                          make_border(row, 0, 0, 2, 2, 9, "constant"))


def test_make_border_refuses_an_unknown_mode():
    with pytest.raises(ValueError):
        make_border(_gray(), 1, 1, 1, 1, 0, "linear_ramp")


# ---------------------------------------------------------------------------
# Watershed
#
# Pixel for pixel identical to `cv2.watershed` across twelve scenes of varied
# size, seed count and texture -- 0 differing pixels in every one. Getting
# there took one correction worth remembering: the active queue level must be
# allowed to move **backwards** when a nearer pixel is pushed. Draining
# strictly forwards segments the same regions but disagrees with OpenCV on
# about one pixel in eighty, all of them on a boundary.

def _two_blobs():
    image = np.full((60, 80, 3), 30, dtype=np.uint8)
    ys, xs = np.mgrid[0:60, 0:80]
    image[((xs - 25) ** 2 + (ys - 30) ** 2) <= 196] = (200, 60, 60)
    image[((xs - 55) ** 2 + (ys - 30) ** 2) <= 196] = (60, 200, 60)

    markers = np.zeros((60, 80), dtype=np.int32)
    markers[30, 25] = 1
    markers[30, 55] = 2
    markers[2, 2] = 3
    return image, markers


def test_watershed_separates_the_seeded_regions():
    image, markers = _two_blobs()
    watershed(image, markers)

    # Every seed keeps its own label and neither blob swallowed the other
    assert markers[30, 25] == 1
    assert markers[30, 55] == 2
    assert 400 < (markers == 1).sum() < 900
    assert 400 < (markers == 2).sum() < 900


def test_watershed_writes_in_place():
    image, markers = _two_blobs()
    before = markers.copy()
    watershed(image, markers)
    assert not np.array_equal(markers, before)


def test_watershed_draws_boundaries_between_regions():
    image, markers = _two_blobs()
    watershed(image, markers)
    assert (markers == -1).any()


def test_the_border_is_watershed_line():
    """OpenCV defines it that way, and it is what keeps the neighbour reads
    inside the image without a bounds test."""
    image, markers = _two_blobs()
    watershed(image, markers)
    assert (markers[0, :] == -1).all()
    assert (markers[-1, :] == -1).all()
    assert (markers[:, 0] == -1).all()
    assert (markers[:, -1] == -1).all()


def test_watershed_leaves_nothing_unclaimed():
    image, markers = _two_blobs()
    watershed(image, markers)
    assert not (markers == 0).any()
    assert not (markers == -2).any()      # the in-queue sentinel must not leak


def test_watershed_wants_markers_the_size_of_the_image():
    image, _ = _two_blobs()
    with pytest.raises(ValueError):
        watershed(image, np.zeros((10, 10), dtype=np.int32))


# ---------------------------------------------------------------------------
# Sub-pixel corners
#
# On a rendered chessboard whose corners are known exactly, seeded with up to
# two pixels of error, this recovers them to 4e-4 of a pixel and agrees with
# `cv2.cornerSubPix` to 6e-4. cv2 is the more precise of the two in absolute
# terms -- 2e-5 -- but both are two orders below the ~0.15 px a real detector
# starts from, which is what the number has to be small against.

def _chessboard(square=40, columns=8, rows=6):
    """A board whose corners sit at the **pixel boundaries**.

    That half pixel matters: the corner between the squares meeting at
    `(i * square, j * square)` is at `i * square - 0.5` in pixel-centre
    coordinates, and forgetting it makes a correct refinement look 0.707 px
    wrong -- which is exactly root two over two.
    """
    image = np.zeros((rows * square, columns * square), dtype=np.uint8)
    for j in range(rows):
        for i in range(columns):
            if (i + j) % 2 == 0:
                image[j * square:(j + 1) * square,
                      i * square:(i + 1) * square] = 255

    image = gaussian_blur(image, 5, 1.2)

    corners = np.array([[i * square - 0.5, j * square - 0.5]
                        for j in range(1, rows) for i in range(1, columns)],
                       dtype=np.float64)
    return image, corners


def test_corner_subpix_recovers_a_displaced_corner():
    image, truth = _chessboard()
    rng = np.random.default_rng(3)
    seeded = truth + rng.uniform(-2.0, 2.0, truth.shape)

    refined = corner_subpix(image, seeded, 5, 5, 40, 0.001)

    before = np.linalg.norm(seeded - truth, axis=1).mean()
    after = np.linalg.norm(refined - truth, axis=1).mean()
    assert before > 1.0
    assert after < 0.01


def test_corner_subpix_leaves_a_settled_corner_alone():
    image, truth = _chessboard()
    refined = corner_subpix(image, truth.copy(), 5, 5, 40, 0.001)
    assert np.linalg.norm(refined - truth, axis=1).max() < 0.01


def test_corner_subpix_returns_the_shape_it_was_given():
    image, truth = _chessboard()
    assert corner_subpix(image, truth.copy()).shape == truth.shape


def test_corner_subpix_wants_pairs():
    image, _ = _chessboard()
    with pytest.raises(ValueError):
        corner_subpix(image, np.zeros((4, 3)))


def test_corner_subpix_wants_a_real_window():
    image, truth = _chessboard()
    with pytest.raises(ValueError):
        corner_subpix(image, truth.copy(), 0, 5)


def test_a_flat_neighbourhood_does_not_move_the_corner():
    """No gradient means no corner; the system is singular and the estimate
    has to stay where it was rather than divide by zero."""
    flat = np.full((40, 40), 128, dtype=np.uint8)
    seeded = np.array([[20.0, 20.0]])
    assert np.array_equal(corner_subpix(flat, seeded.copy()), seeded)


# ----------------------------------------------------------------------------
# Farneback optical flow


def _drifting_pair(shift_x=3, shift_y=2, width=96, height=72):
    """Two crops of one textured field, the second moved by a whole shift.

    Textured because a flow estimator has nothing to measure on flat ground,
    and blurred because the polynomial fit wants a field it can fit -- which
    is also what a real frame looks like once a lens has been through it.
    """
    rng = np.random.default_rng(4)
    field = (rng.random((height + 16, width + 16)) * 255).astype(np.float32)
    field = gaussian_blur(field, 7, 2.0)

    first = np.ascontiguousarray(
        field[8:8 + height, 8:8 + width].astype(np.uint8))
    second = np.ascontiguousarray(
        field[8 + shift_y:8 + shift_y + height,
              8 + shift_x:8 + shift_x + width].astype(np.uint8))

    return first, second


def test_optical_flow_returns_two_planes_of_float():
    first, second = _drifting_pair()
    flow = optical_flow(first, second)
    assert flow.shape == (first.shape[0], first.shape[1], 2)
    assert flow.dtype == np.float32


def test_optical_flow_finds_the_shift_it_was_given():
    """A whole-frame translation of (3, 2) should read as (-3, -2): the flow
    says where a pixel of the first frame went, and a second frame cropped
    three to the right shows the scene having moved three to the left."""
    first, second = _drifting_pair(3, 2)
    flow = optical_flow(first, second)

    middle = flow[20:-20, 20:-20]
    assert abs(np.median(middle[:, :, 0]) + 3.0) < 0.1
    assert abs(np.median(middle[:, :, 1]) + 2.0) < 0.1


def test_optical_flow_of_a_frame_with_itself_is_still():
    """Away from the rim. The rim is **not** still, and that is OpenCV's
    rather than a defect: a pixel on the last row or column has no neighbour
    to interpolate the second frame's fit from, so that fit is taken as zero
    instead of as the first frame's and the difference reads as motion, and
    the coarser pyramid levels then carry a little of it inward.
    `cv2.calcOpticalFlowFarneback` puts the same 0.047 in the same corner of
    the same pair, to within 6e-8, and the same 4e-4 ten pixels in.
    """
    first, _ = _drifting_pair()
    flow = optical_flow(first, first)
    assert np.abs(flow[10:-10, 10:-10]).max() < 1e-3


def test_optical_flow_of_flat_ground_is_still():
    """No gradient means no displacement to measure, and the 1e-3 on the
    determinant is what keeps that from being a division by zero."""
    flat = np.full((64, 64), 120, dtype=np.uint8)
    flow = optical_flow(flat, flat)
    assert np.abs(flow).max() < 1e-6


def test_optical_flow_wants_two_frames_of_a_size():
    first, _ = _drifting_pair()
    with pytest.raises(ValueError):
        optical_flow(first, np.zeros((10, 10), dtype=np.uint8))


def test_optical_flow_wants_a_single_plane():
    with pytest.raises((ValueError, TypeError)):
        optical_flow(np.zeros((32, 32, 3), dtype=np.uint8),
                     np.zeros((32, 32, 3), dtype=np.uint8))


@pytest.mark.parametrize("bad", [
    {"pyr_scale": 1.0},
    {"pyr_scale": 0.0},
    {"levels": -1},
    {"winsize": 0},
    {"poly_n": 0},
])
def test_optical_flow_refuses_a_parameter_out_of_range(bad):
    first, second = _drifting_pair()
    with pytest.raises(ValueError):
        optical_flow(first, second, **bad)


def test_optical_flow_takes_more_levels_than_the_frame_can_hold():
    """The pyramid stops at 32 pixels, so asking for ten levels of a small
    frame builds however many fit rather than shrinking to nothing."""
    first, second = _drifting_pair(3, 2)
    flow = optical_flow(first, second, levels=10)
    middle = flow[20:-20, 20:-20]
    assert abs(np.median(middle[:, :, 0]) + 3.0) < 0.1


# ----------------------------------------------------------------------------
# Corners to track, and following them


def _corner_field(width=160, height=120):
    """A field with corners in it: a grid of squares, blurred."""
    out = np.zeros((height, width), np.uint8)
    out[::20, :] = 200
    out[:, ::20] = 200
    for y in range(10, height, 20):
        for x in range(10, width, 20):
            out[y:y + 6, x:x + 6] = 255
    return gaussian_blur(out, 3, 1.0)


def test_min_eigen_value_is_large_at_a_corner_and_small_along_an_edge():
    image = np.zeros((60, 60), np.uint8)
    image[20:40, 20:40] = 255
    image = gaussian_blur(image, 5, 1.0)

    strength = min_eigen_value(image)

    # The corner of the square against the middle of one of its edges
    assert strength[20, 20] > strength[30, 20] * 5
    assert strength[5, 5] < strength[20, 20] * 0.01


def test_min_eigen_value_is_one_plane_of_float():
    out = min_eigen_value(_corner_field())
    assert out.shape == (120, 160)
    assert out.dtype == np.float32


def test_good_features_finds_the_corners_and_keeps_them_apart():
    corners = good_features_to_track(_corner_field(), max_corners=200,
                                     quality_level=0.01, min_distance=10.0)
    assert corners.shape[1] == 2
    assert len(corners) > 20

    gaps = np.linalg.norm(corners[:, None, :] - corners[None, :, :], axis=2)
    gaps[np.arange(len(corners)), np.arange(len(corners))] = np.inf
    assert gaps.min() >= 10.0


def test_good_features_honours_the_limit():
    corners = good_features_to_track(_corner_field(), max_corners=7,
                                     min_distance=5.0)
    assert len(corners) == 7


def test_good_features_returns_them_strongest_first():
    field = _corner_field()
    corners = good_features_to_track(field, max_corners=20, min_distance=10.0)
    strength = min_eigen_value(field)
    values = [strength[int(y), int(x)] for x, y in corners]
    assert values == sorted(values, reverse=True)


def test_good_features_finds_nothing_on_flat_ground():
    """Every pixel is equally strong, so every pixel ties with its
    neighbours and none of them is a local maximum."""
    flat = np.full((60, 60), 100, np.uint8)
    assert len(good_features_to_track(flat)) == 0


def test_lucas_kanade_follows_a_translation():
    field = _corner_field(200, 160)
    first = np.ascontiguousarray(field[10:150, 10:190])
    second = np.ascontiguousarray(field[13:153, 15:195])   # moved (5, 3)

    points = good_features_to_track(first, max_corners=100,
                                    min_distance=10.0)
    moved, status = lucas_kanade(first, second, points)

    assert moved.shape == points.shape
    assert status.shape == (len(points),)

    followed = status == 1
    assert followed.sum() > len(points) * 0.8

    step = moved[followed] - points[followed]
    assert abs(np.median(step[:, 0]) + 5.0) < 0.2
    assert abs(np.median(step[:, 1]) + 3.0) < 0.2


def test_lucas_kanade_leaves_a_still_frame_alone():
    field = _corner_field()
    points = good_features_to_track(field, max_corners=50, min_distance=10.0)
    moved, status = lucas_kanade(field, field, points)
    assert np.abs(moved[status == 1] - points[status == 1]).max() < 1e-4


def test_lucas_kanade_refuses_a_point_with_no_corner_under_it():
    flat = np.full((80, 80), 100, np.uint8)
    points = np.array([[40.0, 40.0]], np.float32)
    _moved, status = lucas_kanade(flat, flat, points)
    assert status[0] == 0


def test_lucas_kanade_wants_pairs():
    field = _corner_field()
    with pytest.raises(ValueError):
        lucas_kanade(field, field, np.zeros((3, 3), np.float32))


def test_lucas_kanade_wants_two_frames_of_a_size():
    field = _corner_field()
    with pytest.raises(ValueError):
        lucas_kanade(field, np.zeros((10, 10), np.uint8),
                     np.zeros((1, 2), np.float32))


def test_the_pyramid_stops_before_the_window_stops_fitting():
    """A level no bigger than the window is not a level a point can be
    matched on, and OpenCV stops the pyramid before building one. Asking for
    more levels than fit has to give the same answer as asking for exactly
    as many, and on a **periodic** scene it visibly does not if the extra
    level is built: the coarse estimate locks onto the wrong repeat of the
    pattern and the point never finds its way back.
    """
    field = _corner_field(200, 160)
    first = np.ascontiguousarray(field[10:150, 10:190])
    second = np.ascontiguousarray(field[13:153, 15:195])

    points = good_features_to_track(first, max_corners=100,
                                    min_distance=10.0)

    settled, settled_status = lucas_kanade(first, second, points, levels=2)

    for asked in (3, 5, 9):
        moved, status = lucas_kanade(first, second, points, levels=asked)
        assert np.array_equal(status, settled_status)
        assert np.array_equal(moved, settled)


def test_lucas_kanade_does_not_depend_on_how_the_points_were_divided():
    """Following a point reads the pyramid and writes its own two answers, so
    the thread count is a performance knob and nothing else. OpenCV
    parallelises the same loop, and measured on sixteen cores it is three
    times faster than itself on one -- which is why a single threaded port was
    not being compared with like."""
    field = _corner_field(200, 160)
    first = np.ascontiguousarray(field[10:150, 10:190])
    second = np.ascontiguousarray(field[13:153, 15:195])

    points = good_features_to_track(first, max_corners=200, min_distance=6.0)
    settled, settled_status = lucas_kanade(first, second, points, threads=1)

    for count in (2, 3, 8, 0):
        moved, status = lucas_kanade(first, second, points, threads=count)
        assert np.array_equal(status, settled_status)
        assert np.array_equal(moved, settled)


# ----------------------------------------------------------------------------
# The background mixture, and the element OpenCV calls an ellipse


def _noise_sequence(frames=20, height=32, width=40, moving=True):
    rng = np.random.default_rng(5)
    base = rng.random((height, width)) * 100 + 70
    out = []
    for t in range(frames):
        frame = base + rng.integers(-3, 4, (height, width))
        if moving and t >= 6:
            left = 3 + t
            if left + 6 < width:
                frame[10:16, left:left + 6] = 245
        out.append(np.ascontiguousarray(
            np.clip(frame, 0, 255).astype(np.uint8)))
    return out


def test_the_mixture_learns_the_background_and_then_stops_reporting_it():
    model = Mog2Background(history=300, var_threshold=30.0)
    seen = [int((model.apply(f) > 0).sum())
            for f in _noise_sequence(moving=False)]

    # The first frame is all foreground -- there is no background yet
    assert seen[0] == 32 * 40
    assert sum(seen[5:]) == 0


def test_the_mixture_reports_what_moves_through_it():
    model = Mog2Background(history=300, var_threshold=30.0)
    masks = [model.apply(f) for f in _noise_sequence()]
    assert sum(int((m > 0).sum()) for m in masks[6:]) > 0


def test_the_mask_is_one_plane_of_bytes():
    model = Mog2Background()
    mask = model.apply(_noise_sequence(frames=1)[0])
    assert mask.shape == (32, 40)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_the_mixture_counts_its_frames_and_can_forget_them():
    model = Mog2Background()
    for frame in _noise_sequence(frames=4):
        model.apply(frame)
    assert model.frames == 4
    model.reset()
    assert model.frames == 0
    # after forgetting, the next frame is all foreground again
    assert (model.apply(_noise_sequence(frames=1)[0]) > 0).all()


def test_the_mixture_wants_every_frame_the_same_shape():
    model = Mog2Background()
    model.apply(np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError):
        model.apply(np.zeros((9, 8), np.uint8))


def test_a_three_channel_frame_is_accepted():
    model = Mog2Background()
    mask = model.apply(np.zeros((8, 8, 3), np.uint8))
    assert mask.shape == (8, 8)


def test_the_ellipse_element_is_opencvs_not_the_disk():
    """`disk` is VXL's, and rounds an even size down to a symmetric odd one;
    `ellipse` is cv2.MORPH_ELLIPSE, which keeps the even size and puts the
    anchor off centre. The motion detector's default size is 10, so the two
    disagree on the ordinary case rather than on a corner of one."""
    single = np.zeros((41, 41), np.uint8)
    single[20, 20] = 255

    as_disk = dilate(single, 'disk', 10, 10) > 0
    as_ellipse = dilate(single, 'ellipse', 10, 10) > 0

    assert as_disk.sum() != as_ellipse.sum()
    # the ellipse spans the full ten rows and columns asked for
    ys, xs = np.where(as_ellipse)
    assert ys.max() - ys.min() + 1 == 10
    assert xs.max() - xs.min() + 1 == 10
    # where the disk, rounding down, spans nine
    ys, xs = np.where(as_disk)
    assert ys.max() - ys.min() + 1 == 9


def test_an_unknown_element_shape_is_refused():
    with pytest.raises(ValueError):
        dilate(np.zeros((8, 8), np.uint8), 'oval', 3, 3)


def test_resize_uses_opencv_pixel_centres():
    image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    expected = np.array([[0, 64, 191, 255], [64, 96, 159, 191],
                         [191, 159, 96, 64], [255, 191, 64, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(resize(image, 4, 4), expected)
    np.testing.assert_array_equal(resize(image, 4, 4, interpolation="nearest"),
                                  image.repeat(2, 0).repeat(2, 1))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_area_resize_weights_fractional_pixels(dtype):
    image = np.array([[0, 0, 255, 0, 0]], dtype=dtype)
    np.testing.assert_allclose(resize_area(image, 3, 1), [[0, 153, 0]], atol=1e-5)
    np.testing.assert_allclose(resize_area(np.ones((7, 11), dtype=dtype), 4, 3),
                               np.ones((3, 4)), atol=2e-7)


def test_area_enlargement_has_its_own_grid():
    image = np.array([[0, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(resize_area(image, 3, 1), [[0, 128, 255]])


def test_native_filter_releases_the_gil():
    import threading
    import time
    image = np.ones((768, 1024, 3), dtype=np.uint8)
    start = threading.Event()
    finished = threading.Event()
    observations = []

    def observer():
        start.wait()
        # Give the caller time to enter the native operation.
        time.sleep(0.005)
        observations.append(not finished.is_set())

    thread = threading.Thread(target=observer)
    thread.start()
    try:
        start.set()
        gaussian_blur(image, 61)
    finally:
        finished.set()
        thread.join()
    assert observations == [True]


def test_onnx_preprocessing_honours_nearest_interpolation():
    from viame.object_detectors.onnx.onnx_predictor import OnnxPredictor
    predictor = OnnxPredictor.__new__(OnnxPredictor)
    predictor._eval_w = predictor._eval_h = 4
    predictor._interp_name = "nearest"
    predictor._scale, predictor._mean, predictor._std = 1.0, 0.0, 1.0
    image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    expected = image.repeat(2, 0).repeat(2, 1)
    result = predictor._preprocess(image)
    assert result.shape == (1, 3, 4, 4)
    for channel in result[0]:
        np.testing.assert_array_equal(channel, expected)


def test_background_model_serializes_concurrent_updates():
    from concurrent.futures import ThreadPoolExecutor
    model = Mog2Background()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: model.apply(image), range(16)))
    assert model.frames == 16
    assert all(result.shape == image.shape[:2] for result in results)
    model.reset()
    assert model.frames == 0
