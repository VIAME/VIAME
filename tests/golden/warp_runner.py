"""Drive `warp_image` over the golden fixtures.

Shared by the recorder and the golden test. The output is one image, so it
goes through `imageio_utils` like every other image case.
"""

import numpy as np

import warp_cases


def run(impl, variant, source, destination, mask):
    """One warp; returns the resulting image as an array."""
    from kwiver.vital.algo import WarpImage
    from kwiver.vital.types import HomographyD, Image, ImageContainer

    algorithm = WarpImage.create(impl)

    if algorithm is None:
        raise RuntimeError("warp_image '{}' is not registered".format(impl))

    spec = next(entry for entry in warp_cases.WARPS if entry[0] == variant)
    _, homography_name, has_destination, has_mask = spec

    matrix = np.array(warp_cases.HOMOGRAPHIES[homography_name],
                      dtype=np.float64)

    def container(array):
        return ImageContainer(Image(np.ascontiguousarray(array)))

    # The shared mask fixture is boolean, and the OpenCV bridge refuses a
    # boolean image outright -- `vital_to_ocv` has no `cv::Mat` type for one.
    # A byte mask is what every VIAME detector produces, so that is what the
    # alpha cases pass: the implementation divides by 255 for an 8U mask.
    if mask is not None and mask.dtype == np.bool_:
        mask = (mask.astype(np.uint8) * 255)

    warped = algorithm.warp(
        container(source),
        container(destination) if has_destination else None,
        HomographyD(matrix),
        container(mask) if has_mask else None)

    return warped.asarray()
