"""
ckwg +31
Copyright 2016 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

vital::algo::estimate_similarity_transform interface

"""
import ctypes

from vital.algo import VitalAlgorithm
from vital.exceptions.algorithm import VitalAlgorithmException
from vital.types import (
    CameraMap,
    EigenArray,
    LandmarkMap,
    Similarity,
)


class EstimateSimilarityTransform (VitalAlgorithm):

    TYPE_NAME = 'estimate_similarity_transform'

    def estimate_from_points(self, from_pts, to_pts):
        """
        Estimate the similarity transform between two corresponding point sets

        :raises VitalAlgorithmException: from and to point sets are misaligned,
            insufficient or degenerate

        :param from_pts: Iterable of 3D points in the ``from`` space.
        :type from_pts: collections.Iterable[EigenArray | collections.Sequence[float]]

        :param to_pts: Iterable of 3D points in the ``to`` space.
        :type to_pts: collections.Iterable[EigenArray | collections.Sequence[float]]

        :return: New similarity instance
        :rtype: Similarity

        """
        ea_type = EigenArray.c_ptr_type(3, 1, ctypes.c_double)

        # make C arrays from input points
        from_list = [EigenArray.from_iterable(p, target_shape=(3, 1))
                     for p in from_pts]
        to_list = [EigenArray.from_iterable(p, target_shape=(3, 1))
                   for p in to_pts]
        if len(from_list) != len(to_list):
            raise VitalAlgorithmException(
                "From and to iterables not the same length: %d != %d"
                % (len(from_list), len(to_list))
            )
        n = len(from_list)
        from_ptr_arr = (ea_type * n)()
        for i, e in enumerate(from_list):
            from_ptr_arr[i] = e.c_pointer

        to_ptr_arr = (ea_type * n)()
        for i, e in enumerate(to_list):
            to_ptr_arr[i] = e.c_pointer

        sim_ptr = self._call_cfunc(
            'vital_algorithm_estimate_similarity_transform_estimate_transform_points',
            [self.C_TYPE_PTR, ctypes.c_size_t, ctypes.POINTER(ea_type),
             ctypes.POINTER(ea_type)],
            [self, n, from_ptr_arr, to_ptr_arr],
            Similarity.c_ptr_type(ctypes.c_double),
            {
                1: VitalAlgorithmException
            }
        )
        return Similarity(ctype=ctypes.c_double, from_cptr=sim_ptr)

    def estimate_from_camera_maps(self, from_cm, to_cm):
        """
        Estimate the similarity transform between two corresponding camera maps

        Cameras with corresponding frame IDs in the two maps are paired for
        transform estimation. Cameras with no corresponding frame ID in the
        other map are ignored. An exception is set if there are no shared frame
        IDs between the two provided maps (nothing to pair).

        :param from_cm: Map of original cameras, sharing N frames with the
            transformed cameras, where N > 0.
        :type from_cm: CameraMap

        :param to_cm:  Map of transformed cameras, sharing N frames with the
            original cameras, where N > 0.
        :type to_cm: CameraMap

        :return: New estimated similarity transform mapping camera centers in
            the ``from`` space to camera centers in the ``to`` space.
        :rtype: Similarity

        """
        cptr = self._call_cfunc(
            'vital_algorithm_estimate_similarity_transform_estimate_camera_map',
            [self.C_TYPE_PTR, CameraMap.c_ptr_type(), CameraMap.c_ptr_type()],
            [self, from_cm, to_cm],
            Similarity.c_ptr_type(ctypes.c_double),
            {
                1: VitalAlgorithmException
            }
        )
        return Similarity(ctype=ctypes.c_double, from_cptr=cptr)

    def estimate_from_landmark_maps(self, from_lm, to_lm):
        """
        Estimate the similarity transform between two corresponding landmark
        maps.

        Landmarks with corresponding frame IDs in the two maps are paired for
        transform estimation. Landmarks with no corresponding frame ID in the
        other map are ignored. An exception is set if there are no shared frame
        IDs between the two provided maps (nothing to pair).

        :param from_lm: Map of original landmarks, sharing N frames with the
            transformed landmarks, where N > 0.
        :type from_lm: LandmarkMap

        :param to_lm: Map of transformed landmarks, sharing N frames with the
            original landmarks, where N > 0.
        :type to_lm: LandmarkMap

        :return: An estimated similarity transform mapping landmark centers in
            the ``from`` space to camera centers in the ``to`` space.
        :rtype: Similarity

        """
        cptr = self._call_cfunc(
            'vital_algorithm_estimate_similarity_transform_estimate_landmark_map',
            [self.C_TYPE_PTR, LandmarkMap.c_ptr_type(), LandmarkMap.c_ptr_type()],
            [self, from_lm, to_lm],
            Similarity.c_ptr_type(ctypes.c_double),
            {
                1: VitalAlgorithmException
            }
        )
        return Similarity(ctype=ctypes.c_double, from_cptr=cptr)
