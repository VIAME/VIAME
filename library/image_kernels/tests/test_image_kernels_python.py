"""The python bindings run the same kernels the C++ pipelines run.

That is the whole point of them: a frame resized in python and a frame
resized in a pipeline should agree. Measured against cv2 when written, on a
natural frame:

    to_gray        max difference 0   -- bit identical
    swap_channels  identical
    crop           identical
    resize         max difference 25  -- a different pixel centre convention
    to_hsv/to_hls  max difference 1
    to_lab         max difference 2

The three colour spaces round trip at least as well as OpenCV's own do: on a
random frame, 4 against its 5 for HSV and HLS, and 21 for both on L*a*b*,
where the loss is the 8-bit quantisation rather than either implementation.

The resize difference is expected and is why these exist rather than Pillow:
matching the C++ half matters, matching OpenCV does not, and OpenCV is what
is being removed.
"""
import numpy as np
import pytest

from viame.image_kernels import (add_weighted, approx_poly, arc_length,
                                 bounding_rect,
                                 box_blur, clahe, contour_area, convex_hull,
                                 crop, distance_transform,
                                 intersect_convex, moments,
                                 demosaic, dilate, draw_circle, draw_line,
                                 find_contours, label_components,
                                 corner_subpix, make_border,
                                 match_template, morphology,
                                 watershed,
                                 min_area_rect,
                                 draw_rect, draw_text, equalize, erode,
                                 fill_polygon,
                                 from_hls, from_hsv, from_lab, gaussian_blur,
                                 normalize, remap, resize, resize_area,
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
# all **bit identical**, and clahe is within one grey level. Those are strong
# enough agreements to assert shape and invariants here and leave the pixel
# comparison to tests/golden.

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
