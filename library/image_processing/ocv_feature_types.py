# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Conversions between vital's feature types and OpenCV's, for python.

This is `library/opencv_bridge/{feature_set,descriptor_set,match_set}` in
python, and it keeps their behaviour rather than improving on it, because
`tests/golden/opencv` holds the ported implementations to what the C++ ones
produced.

The part that matters most is `OCVFeatureSet`. `vital::feature` carries a
location, a scale, an angle and a magnitude, and a `cv::KeyPoint` carries
those plus an octave -- and a SIFT descriptor computed for a keypoint whose
octave has been lost is computed at the wrong scale, silently. The C++ bridge
solved that by having its own `ocv::feature_set` hold the keypoints and by
having the extractor `dynamic_cast` to it, falling back to a conversion when
the set came from somewhere else. This is that, in python: a feature set
subclass that carries the keypoints, and an `isinstance` where the C++ had
its cast.
"""

import numpy as np

from kwiver.vital.types import DescriptorSet, FeatureF, FeatureSet, MatchSet
from kwiver.vital.types.descriptor import new_descriptor


class OCVFeatureSet(FeatureSet):
    """A vital feature set that also carries the cv2 keypoints it came from.

    `library/opencv_bridge/feature_set.h` in python. `features()` builds the
    vital view on demand, exactly as the C++ one does, so a caller that only
    wants locations pays nothing for the keypoints riding along.
    """

    def __init__(self, keypoints):
        FeatureSet.__init__(self)
        self.keypoints = list(keypoints)
        self._features = None

    def size(self):
        return len(self.keypoints)

    def features(self):
        if self._features is None:
            self._features = [keypoint_to_feature(kp) for kp in self.keypoints]

        return self._features


def keypoint_to_feature(keypoint):
    """One `cv::KeyPoint` as a `vital::feature_f`.

    The same four fields the C++ bridge copies: location, response as the
    magnitude, size as the scale, and the angle. `octave` and `class_id` have
    nowhere to go, which is what `OCVFeatureSet` is for.
    """
    return FeatureF(
        loc=np.array([float(keypoint.pt[0]), float(keypoint.pt[1])]),
        mag=float(keypoint.response),
        scale=float(keypoint.size),
        angle=float(keypoint.angle))


def features_to_keypoints(features):
    """A feature set as `cv::KeyPoint`s, keeping the octave when it is there.

    The `isinstance` is the C++ bridge's `dynamic_cast`: a set this module
    produced hands back the keypoints it was built from, octave included, and
    anything else is converted field by field with the octave left at zero --
    which is what `features_to_ocv_keypoints` does and what it costs.
    """
    import cv2

    if isinstance(features, OCVFeatureSet):
        return list(features.keypoints)

    return [cv2.KeyPoint(x=float(feature.location[0]),
                         y=float(feature.location[1]),
                         size=float(feature.scale),
                         angle=float(feature.angle),
                         response=float(feature.magnitude))
            for feature in features.features()]


def descriptors_to_set(matrix):
    """An (n, d) cv2 descriptor matrix as a `vital::descriptor_set`.

    Float descriptors become `descriptor_fixed<float, d>`, which is what the
    C++ bridge builds for a `CV_32F` matrix, and a binary descriptor's bytes
    become one byte per element the same way.
    """
    if matrix is None or len(matrix) == 0:
        return DescriptorSet([])

    matrix = np.asarray(matrix)
    ctype = "b" if matrix.dtype == np.uint8 else "f"
    width = matrix.shape[1]

    out = []
    for row in matrix:
        descriptor = new_descriptor(width, ctype)
        for index in range(width):
            descriptor[index] = row[index]
        out.append(descriptor)

    return DescriptorSet(out)


def set_to_descriptors(descriptor_set, binary=False):
    """A `vital::descriptor_set` as the (n, d) matrix cv2 wants.

    `binary` because the FLANN matcher's LSH index needs `CV_8U` where
    everything else here is `CV_32F`; the C++ bridge chose by the descriptor's
    own type, which python cannot see.
    """
    if descriptor_set is None or descriptor_set.size() == 0:
        return None

    rows = [np.asarray(d.todoublearray()) for d in descriptor_set.descriptors()]

    return np.vstack(rows).astype(np.uint8 if binary else np.float32)


def matches_to_set(matches):
    """A list of `cv::DMatch` as a `vital::match_set` of index pairs."""
    return MatchSet([(int(m.queryIdx), int(m.trainIdx)) for m in matches])
