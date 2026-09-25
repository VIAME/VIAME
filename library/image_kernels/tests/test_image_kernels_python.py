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

from viame.image_kernels import (add_weighted, approx_poly, bounding_rect,
                                 box_blur, clahe, contour_area, convex_hull,
                                 crop,
                                 demosaic, dilate, draw_circle, draw_line,
                                 find_contours, label_components,
                                 make_border, match_template, morphology,
                                 watershed,
                                 min_area_rect,
                                 draw_rect, draw_text, equalize, erode,
                                 fill_polygon,
                                 from_hls, from_hsv, from_lab, gaussian_blur,
                                 normalize, remap, resize, resize_area,
                                 swap_channels, text_size, to_gray, to_hls,
                                 to_hsv, to_lab, to_rgb, warp_affine,
                                 warp_perspective)


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
# Hue is on OpenCV's 0..179 scale, not 0..360, because that is what the eight
# python call sites that used `cv2.COLOR_RGB2HSV` were written against.

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


def test_an_unknown_border_is_refused():
    with pytest.raises(ValueError):
        gaussian_blur(_gray(), 5, 0.0, "wrap")


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
