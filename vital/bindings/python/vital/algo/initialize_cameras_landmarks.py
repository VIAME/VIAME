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

vital::algo::initialize_cameras_landmarks interface

"""
import ctypes

from vital.algo import VitalAlgorithm

from vital.types import (
    CameraMap,
    LandmarkMap,
    TrackSet,
)


class InitializeCamerasLandmarks (VitalAlgorithm):

    TYPE_NAME = "initialize_cameras_landmarks"

    def initialize(self, cmap, lmap, tset):
        """
        Initialize the camera and landmark parameters given a set of tracks

        :param cmap: Cameras to initialize
        :type cmap: CameraMap

        :param lmap: Landmarks to initialize
        :type lmap: LandmarkMap

        :param tset: Tracks to use as constraints
        :type tset: TrackSet

        :return: New, initialized camera and landmark maps.
        :rtype: (CameraMap, LandmarkMap)

        """
        # make a separate copy of pointer container in prep for passing by ref
        cmap_ptr = CameraMap.c_ptr_type()(cmap.c_pointer.contents)
        lmap_ptr = LandmarkMap.c_ptr_type()(lmap.c_pointer.contents)

        self._call_cfunc(
            "vital_algorithm_initialize_cameras_landmarks_initialize",
            [self.C_TYPE_PTR,
             ctypes.POINTER(CameraMap.c_ptr_type()),
             ctypes.POINTER(LandmarkMap.c_ptr_type()),
             TrackSet.c_ptr_type()],
            [self, ctypes.byref(cmap_ptr), ctypes.byref(lmap_ptr), tset]
        )

        # Initialize new objects if "returned" pointers are different from input
        # objects
        r_cmap = cmap
        if ctypes.addressof(cmap_ptr.contents) != ctypes.addressof(cmap.c_pointer.contents):
            self._log.debug("Creating new CameraMap instance")
            r_cmap = CameraMap(from_cptr=cmap_ptr)
        r_lmap = lmap
        if ctypes.addressof(lmap_ptr.contents) != ctypes.addressof(lmap.c_pointer.contents):
            self._log.debug("Creating new LandmarkMap instance")
            r_lmap = LandmarkMap(from_cptr=lmap_ptr)

        return r_cmap, r_lmap
