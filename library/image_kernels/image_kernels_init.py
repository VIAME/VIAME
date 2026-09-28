"""VIAME's own image kernels.

The operations python used OpenCV for, running the same code the C++
pipelines run. `viame.utilities.imageops` is the friendlier face of this for
reading and writing files; this is the pixel work.
"""

from viame.image_kernels._image_kernels import (  # noqa: F401
    GaussianWorkspace,
    StereoWorkspace,
    kernel_thread_count,
    set_kernel_thread_count,
    add_weighted,
    approx_poly,
    arc_length,
    bilateral_blur,
    bounding_rect,
    box_blur,
    canny,
    clahe,
    contour_area,
    convex_hull,
    corner_subpix,
    crop,
    demosaic,
    dilate,
    distance_transform,
    draw_circle,
    draw_line,
    draw_polyline,
    draw_rect,
    draw_text,
    equalize,
    erode,
    fill_ellipse,
    filter_disparity_wls,
    fill_polygon,
    filter_2d,
    find_borders,
    find_contours,
    from_hls,
    from_luv,
    from_xyz,
    from_ycrcb,
    from_hsv,
    denoise,
    denoise_colour,
    fast_corners,
    from_lab,
    gaussian_blur,
    gaussian_blur_float_taps,
    histogram,
    hough_circles,
    good_features_to_track,
    grab_cut,
    intersect_convex,
    label_components,
    laplacian,
    lucas_kanade,
    make_border,
    match_template,
    min_area_rect,
    min_eigen_value,
    moments,
    Mog2Background,
    morphology,
    mean_shift_blur,
    median_blur,
    normalize,
    pyramid_down,
    pyramid_up,
    optical_flow,
    remap as _remap,
    resize,
    resize_area,
    resize_letterbox,
    smooth_globally,
    stereo_bm,
    stereo_sgbm,
    swap_channels,
    text_size,
    to_gray,
    to_hls,
    to_luv,
    to_xyz,
    to_ycrcb,
    to_hsv,
    to_lab,
    to_rgb,
    warp_affine as _warp_affine,
    warp_polar,
    warp_perspective as _warp_perspective,
    watershed,
)

__all__ = [
    "add_weighted",
    "approx_poly",
    "arc_length",
    "bilateral_blur",
    "bounding_rect",
    "box_blur",
    "canny",
    "clahe",
    "contour_area",
    "convex_hull",
    "corner_subpix",
    "crop",
    "demosaic",
    "dilate",
    "distance_transform",
    "draw_circle",
    "draw_line",
    "draw_polyline",
    "draw_rect",
    "draw_text",
    "equalize",
    "erode",
    "fill_ellipse",
    "filter_disparity_wls",
    "fill_polygon",
    "filter_2d",
    "find_borders",
    "find_contours",
    "from_hls",
    "from_luv",
    "from_xyz",
    "from_ycrcb",
    "from_hsv",
    "denoise",
    "denoise_colour",
    "fast_corners",
    "from_lab",
    "gaussian_blur",
    "gaussian_blur_float_taps",
    "histogram",
    "hough_circles",
    "good_features_to_track",
    "grab_cut",
    "intersect_convex",
    "component_stats",
    "label_components",
    "laplacian",
    "lucas_kanade",
    "make_border",
    "match_template",
    "min_area_rect",
    "min_eigen_value",
    "moments",
    "Mog2Background",
    "morphology",
    "mean_shift_blur",
    "median_blur",
    "normalize",
    "pyramid_down",
    "pyramid_up",
    "optical_flow",
    "remap",
    "resize",
    "set_kernel_thread_count",
    "resize_area",
    "resize_letterbox",
    "smooth_globally",
    "stereo_bm",
    "stereo_sgbm",
    "swap_channels",
    "text_size",
    "to_gray",
    "to_hls",
    "to_luv",
    "to_xyz",
    "to_ycrcb",
    "to_hsv",
    "to_lab",
    "to_rgb",
    "warp_affine",
    "warp_polar",
    "warp_perspective",
    "watershed",
]


# ---------------------------------------------------------------------------
# Per-plane border constants
#
# `cv2.warpAffine`'s `borderValue` takes a scalar **or one value per channel**,
# and the siammask crops use the second form: they pad with the image's
# channel means so a crop that runs off the edge does not get a black border
# the network has never seen. The kernels take one constant for the whole
# image, which is the right shape for C++ where the planes are warped
# together, so the sequence case is unrolled here rather than pushed down.

import numpy as _np  # noqa: E402


def _per_plane(kernel, image, constant, *args, **kwargs):
    """Run `kernel` once per plane when `constant` gives one value per plane."""
    if _np.isscalar(constant):
        return kernel(image, *args, constant=float(constant), **kwargs)

    constants = [float(value) for value in constant]

    if image.ndim != 3:
        if len(set(constants)) != 1:
            raise ValueError(
                "a single plane image takes one border constant, got "
                "{}".format(len(constants)))
        return kernel(image, *args, constant=constants[0], **kwargs)

    if len(constants) != image.shape[2]:
        raise ValueError(
            "wanted one border constant per plane: {} planes, {} "
            "constants".format(image.shape[2], len(constants)))

    planes = [kernel(_np.ascontiguousarray(image[:, :, plane]), *args,
                     constant=value, **kwargs)
              for plane, value in enumerate(constants)]

    return _np.stack(planes, axis=-1)


def component_stats(labels, count):
    """`(stats, centroids)` for a `label_components` labelling.

    `stats` is one row per label of `left, top, width, height, area`, and
    `centroids` the mean x and y of each. Label 0 is the background.

    **Do not read meaning into a label's number.** `label_components`
    partitions a mask the same way every time, but which component gets which
    number depends on the scan, so a caller holding a recorded label value
    would be disappointed; one asking which component is the largest is safe.
    """
    labels = _np.asarray(labels)
    height, width = labels.shape[:2]
    flat = labels.reshape(-1)

    rows = _np.repeat(_np.arange(height), width)
    columns = _np.tile(_np.arange(width), height)

    areas = _np.bincount(flat, minlength=count)
    sums_x = _np.bincount(flat, weights=columns.astype(_np.float64),
                          minlength=count)
    sums_y = _np.bincount(flat, weights=rows.astype(_np.float64),
                          minlength=count)

    # One pass for each extreme rather than one pass per label: a mask can
    # hold thousands of components and `labels == n` in a loop is quadratic.
    left = _np.full(count, width, dtype=_np.int64)
    top = _np.full(count, height, dtype=_np.int64)
    right = _np.full(count, -1, dtype=_np.int64)
    bottom = _np.full(count, -1, dtype=_np.int64)
    _np.minimum.at(left, flat, columns)
    _np.minimum.at(top, flat, rows)
    _np.maximum.at(right, flat, columns)
    _np.maximum.at(bottom, flat, rows)

    present = areas > 0
    stats = _np.zeros((count, 5), dtype=_np.int32)
    stats[present, 0] = left[present]
    stats[present, 1] = top[present]
    stats[present, 2] = (right - left + 1)[present]
    stats[present, 3] = (bottom - top + 1)[present]
    stats[:, 4] = areas

    centroids = _np.zeros((count, 2), dtype=_np.float64)

    with _np.errstate(invalid="ignore", divide="ignore"):
        centroids[:, 0] = _np.where(present, sums_x / areas, 0.0)
        centroids[:, 1] = _np.where(present, sums_y / areas, 0.0)

    return stats, centroids


def remap(image, map_x, map_y, interpolation="bilinear", border="constant",
          constant=0.0):
    """cv2.remap. See `_image_kernels.remap`; `constant` may be per plane."""
    return _per_plane(_remap, image, constant, map_x, map_y,
                      interpolation=interpolation, border=border)


def warp_affine(image, transform, width=0, height=0,
                interpolation="bilinear", border="constant", constant=0.0,
                inverse=False):
    """cv2.warpAffine. `constant` may be a scalar or one value per plane.

    `inverse` takes the transform as already mapping destination to source,
    which is cv2.WARP_INVERSE_MAP.
    """
    return _per_plane(_warp_affine, image, constant, transform,
                      width=width, height=height,
                      interpolation=interpolation, border=border,
                      inverse=inverse)


def warp_perspective(image, transform, width=0, height=0,
                     interpolation="bilinear", border="constant",
                     constant=0.0, inverse=False):
    """cv2.warpPerspective. `constant` may be a scalar or one per plane."""
    return _per_plane(_warp_perspective, image, constant, transform,
                      width=width, height=height,
                      interpolation=interpolation, border=border,
                      inverse=inverse)
