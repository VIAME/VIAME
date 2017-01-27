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

vital::algo::estimate_canonical_transform interface

"""
import ctypes

from vital.algo import VitalAlgorithm
from vital.exceptions.algorithm import VitalAlgorithmException
from vital.types import (
    CameraMap,
    LandmarkMap,
    Similarity,
)


class EstimateCanonicalTransform (VitalAlgorithm):

    TYPE_NAME = 'estimate_canonical_transform'

    def estimate(self, cameras, landmarks):
        """
        Estimate a canonical similarity transform for cameras and points

        :param cameras: The camera map containing all the cameras
        :type cameras: CameraMap

        :param landmarks: The landmark map containing all the 3D landmarks
        :type landmarks: LandmarkMap

        :return: New estimated similarity transformation mapping the data to the
            canonical space.
        :rtype: Similarity

        """
        cptr = self._call_cfunc(
            'vital_algorithm_estimate_canonical_transform_estimate',
            [self.C_TYPE_PTR, CameraMap.c_ptr_type(), LandmarkMap.c_ptr_type()],
            [self, cameras, landmarks],
            Similarity.c_ptr_type(ctypes.c_double),
            {
                1: VitalAlgorithmException
            }
        )
        return Similarity(ctype=ctypes.c_double, from_cptr=cptr)
