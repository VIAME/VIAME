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

vital::algo::triangulate_landmarks interface

"""
import ctypes

from vital.algo import VitalAlgorithm

from vital.types import (
    CameraMap,
    LandmarkMap,
    TrackSet,
)


class TriangulateLandmarks (VitalAlgorithm):

    TYPE_NAME = "triangulate_landmarks"

    def triangulate(self, cameras, tracks, landmarks):
        """
        Triangulate the landmark locations given sets of cameras and tracks

        This function only triangulates the landmarks with indices in the
        landmark map and which have support in the tracks and cameras

        :param cameras: cameras viewing the landmarks
        :type cameras: CameraMap

        :param tracks: tracks to use as constraints
        :type tracks: TrackSet

        :param landmarks: landmarks to triangulate
        :type landmarks: LandmarkMap

        :return: New landmarks instance of triangulated landmarks

        """
        # Copy pointer container for reference updating so we don't pollute the
        # input instance.
        lmap_ptr = LandmarkMap.c_ptr_type()(landmarks.c_pointer.contents)

        self._call_cfunc(
            'vital_algorithm_triangulate_landmarks_triangulate',
            [self.C_TYPE_PTR, CameraMap.c_ptr_type(), TrackSet.c_ptr_type(),
             ctypes.POINTER(LandmarkMap.c_ptr_type())],
            [self, cameras, tracks, ctypes.byref(landmarks)]
        )

        r_lmap = landmarks
        if ctypes.addressof(lmap_ptr.contents) != ctypes.addressof(landmarks.c_pointer.contents):
            r_lmap = LandmarkMap(from_cptr=lmap_ptr)

        return r_lmap
