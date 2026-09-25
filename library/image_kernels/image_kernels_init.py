"""VIAME's own image kernels.

The operations python used OpenCV for, running the same code the C++
pipelines run. `viame.utilities.imageops` is the friendlier face of this for
reading and writing files; this is the pixel work.
"""

from viame.image_kernels._image_kernels import (  # noqa: F401
    add_weighted,
    approx_poly,
    arc_length,
    bounding_rect,
    box_blur,
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
    fill_polygon,
    filter_2d,
    find_borders,
    find_contours,
    from_hls,
    from_hsv,
    from_lab,
    gaussian_blur,
    good_features_to_track,
    intersect_convex,
    label_components,
    lucas_kanade,
    make_border,
    match_template,
    min_area_rect,
    min_eigen_value,
    moments,
    morphology,
    normalize,
    optical_flow,
    remap as _remap,
    resize,
    resize_area,
    swap_channels,
    text_size,
    to_gray,
    to_hls,
    to_hsv,
    to_lab,
    to_rgb,
    warp_affine as _warp_affine,
    warp_perspective as _warp_perspective,
    watershed,
)

__all__ = [
    "add_weighted",
    "approx_poly",
    "arc_length",
    "bounding_rect",
    "box_blur",
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
    "fill_polygon",
    "filter_2d",
    "find_borders",
    "find_contours",
    "from_hls",
    "from_hsv",
    "from_lab",
    "gaussian_blur",
    "good_features_to_track",
    "intersect_convex",
    "label_components",
    "lucas_kanade",
    "make_border",
    "match_template",
    "min_area_rect",
    "min_eigen_value",
    "moments",
    "morphology",
    "normalize",
    "optical_flow",
    "remap",
    "resize",
    "resize_area",
    "swap_channels",
    "text_size",
    "to_gray",
    "to_hls",
    "to_hsv",
    "to_lab",
    "to_rgb",
    "warp_affine",
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


def remap(image, map_x, map_y, interpolation="bilinear", border="constant",
          constant=0.0):
    """cv2.remap. See `_image_kernels.remap`; `constant` may be per plane."""
    return _per_plane(_remap, image, constant, map_x, map_y,
                      interpolation=interpolation, border=border)


def warp_affine(image, transform, width=0, height=0,
                interpolation="bilinear", border="constant", constant=0.0):
    """cv2.warpAffine. `constant` may be a scalar or one value per plane."""
    return _per_plane(_warp_affine, image, constant, transform,
                      width=width, height=height,
                      interpolation=interpolation, border=border)


def warp_perspective(image, transform, width=0, height=0,
                     interpolation="bilinear", border="constant",
                     constant=0.0):
    """cv2.warpPerspective. `constant` may be a scalar or one per plane."""
    return _per_plane(_warp_perspective, image, constant, transform,
                      width=width, height=height,
                      interpolation=interpolation, border=border)
