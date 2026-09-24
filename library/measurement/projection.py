# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""VIAME's camera geometry, the same code the C++ pipelines run.

What python used `cv2.projectPoints`, `cv2.undistortPoints`,
`cv2.Rodrigues`, `cv2.stereoRectify` and `cv2.initUndistortRectifyMap` for.
The implementations are `library/measurement/projection.h`; this is the name
python calls them by.

`rectification_maps` gives the two maps `image_kernels.remap` samples
through, so a rectification is those two calls and nothing else.
"""

from viame.measurement._projection import (  # noqa: F401
    project_points,
    rectification_maps,
    rodrigues,
    stereo_rectify,
    undistort_points,
)

__all__ = [
    "project_points",
    "rectification_maps",
    "rodrigues",
    "stereo_rectify",
    "undistort_points",
]
