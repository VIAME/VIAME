# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""VIAME's own SIFT, SURF and ORB, as functions of an array.

What python used `cv2.SIFT_create().detectAndCompute` and
`cv2.xfeatures2d.SURF_create` for. The implementations are
`library/image_processing/sift.h` and `surf.h` -- the same code the
`ocv_SIFT` and `ocv_SURF` algorithms run -- and this is the name a script
calls them by.

Keypoints are one **(n, 6) float32 array**: x, y, size, angle, response, and
the packed octave. The last is SIFT's and says which level of the Gaussian
pyramid the keypoint was found at, so `sift_describe` samples the right one;
it is zero for SURF, which has no such field, and zero for keypoints a caller
invented, which describes them at the base of the pyramid. That is what cv2
does with the same input. ORB puts its **pyramid level** in that column
rather than a packed octave, as cv2's ORB does, and its descriptors are
uint8 rather than float: 32 bytes of bit comparisons, to be matched by
Hamming distance (`matching.ratio_match( ..., binary=True )`).

ORB's keypoints come back in detection order. cv2's come back in whatever
order `std::nth_element` left them in, which its own standard library does
not specify; the set is the same either way, and a matcher pairs row i of
the descriptors with keypoint i whichever order that is.

Reading a column is how a caller uses these::

    keypoints, descriptors = features.sift(grey)
    xy = keypoints[:, :2]
"""

from viame.image_processing._features import (  # noqa: F401
    orb,
    orb_describe,
    sift,
    sift_describe,
    surf,
)

__all__ = [
    "orb",
    "orb_describe",
    "sift",
    "sift_describe",
    "surf",
]
