"""
ckwg +31
Copyright 2015-2016 by Kitware, Inc.
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

Interface to vital::camera_map class.

"""
# -*- coding: utf-8 -*-
__author__ = 'paul.tunison@kitware.com'

import ctypes

from vital.types import Camera
from vital.util import VitalObject, VitalErrorHandle


class CameraMap (VitalObject):
    """ vital::camera_map interface class """

    def __init__(self, frame2cam_map, from_cptr=None):
        """
        :param frame2cam_map: Association of frame number to camera instance
        :type frame2cam_map: dict[int, vital.types.Camera]

        """
        super(CameraMap, self).__init__(from_cptr, frame2cam_map)

    def _new(self, frame2cam_map):
        """
        :type frame2cam_map: dict[int, vital.types.Camera]
        """
        cm_new = self.VITAL_LIB.vital_camera_map_new
        cm_new.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64),
                           ctypes.POINTER(Camera.C_TYPE_PTR)]
        cm_new.restype = self.C_TYPE_PTR

        # Construct input frame and camera arrays
        fn_list = []
        cam_list = []
        for fn, c in frame2cam_map.iteritems():
            fn_list.append(fn)
            cam_list.append(c)
        fn_array_t = ctypes.c_int64 * len(frame2cam_map)
        cam_array_t = Camera.C_TYPE_PTR * len(frame2cam_map)
        # noinspection PyCallingNonCallable
        c_fn_array = fn_array_t(*fn_list)
        # noinspection PyCallingNonCallable,PyProtectedMember
        c_cam_array = cam_array_t(*(c.c_pointer for c in cam_list))

        return cm_new(len(frame2cam_map), c_fn_array, c_cam_array)

    def _destroy(self):
        cm_del = self.VITAL_LIB.vital_camera_map_destroy
        cm_del.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        with VitalErrorHandle() as eh:
            cm_del(self, eh)

    @property
    def size(self):
        """
        :return: Number of elements in this mapping.
        :rtype: int
        """
        cm_size = self.VITAL_LIB.vital_camera_map_size
        cm_size.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        cm_size.restype = ctypes.c_size_t
        with VitalErrorHandle() as eh:
            return cm_size(self, eh)

    def to_dict(self):
        """
        :return: Internal frame-number to cameras mapping
        :rtype: dict[int, Camera]
        """
        cm_get_map = self.VITAL_LIB['vital_camera_map_get_map']
        cm_get_map.argtypes = [self.c_ptr_type(),
                               ctypes.POINTER(ctypes.c_size_t),
                               ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                               ctypes.POINTER(ctypes.POINTER(Camera.c_ptr_type())),
                               VitalErrorHandle.c_ptr_type()]

        length = ctypes.c_size_t()
        frame_numbers = ctypes.POINTER(ctypes.c_int64)()
        cameras = ctypes.POINTER(Camera.c_ptr_type())()

        with VitalErrorHandle() as eh:
            cm_get_map(self,
                       ctypes.byref(length),
                       ctypes.byref(frame_numbers),
                       ctypes.byref(cameras),
                       eh)

        m = {}
        for i in xrange(length.value):
            # copy camera cptr so we don't
            cptr = Camera.c_ptr_type()(cameras[i].contents)
            m[frame_numbers[i]] = Camera(from_cptr=cptr)

        # Free frame number and camera pointer arrays
        free_ptr = self.VITAL_LIB['vital_free_pointer']
        free_ptr(frame_numbers)
        free_ptr(cameras)

        return m
