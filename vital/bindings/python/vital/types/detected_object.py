"""
ckwg +31
Copyright 2017 by Kitware, Inc.
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

Interface to VITAL detected_object class.

"""
import ctypes

from vital.util import VitalObject
from vital.util import VitalErrorHandle

from vital.types import BoundingBox
from vital.types import DetectedObjectType


class DetectedObject (VitalObject):
    """
    vital::detected_object interface class
    """

    def __init__(self, bbox=None, confid=None, tot=None, from_cptr=None):
        """
        Create a simple detected object type

         """
        super(DetectedObject, self).__init__(from_cptr, bbox, confid, tot)

    def _new(self, bbox, confid, tot):
        do_new = self.VITAL_LIB.vital_detected_object_new_with_bbox
        do_new.argtypes = [BoundingBox.C_TYPE_PTR, ctypes.c_double, DetectedObjectType.C_TYPE_PTR]
        do_new.restype = self.C_TYPE_PTR
        return do_new(bbox, ctypes.c_double( confid ), tot)

    def _destroy(self):
        do_del = self.VITAL_LIB.vital_detected_object_destroy
        do_del.argtypes = [self.C_TYPE_PTR]
        do_del(self)

    def bounding_box(self):
        # Get C pointer to internal bounding box
        do_get_bb = self.VITAL_LIB.vital_detected_object_bounding_box
        do_get_bb.argtypes = [self.C_TYPE_PTR]
        do_get_bb.restype = BoundingBox.C_TYPE_PTR
        bb_c_ptr = do_get_bb(self)
        # Make copy of bounding box to return
        do_bb_cpy = self.VITAL_LIB.vital_bounding_box_copy
        do_bb_cpy.argtypes = [BoundingBox.C_TYPE_PTR]
        do_bb_cpy.restype = BoundingBox.C_TYPE_PTR
        return BoundingBox( from_cptr=do_bb_cpy( bb_c_ptr ) )

    def set_bounding_box(self, bbox):
        do_sbb = self.VITAL_LIB.vital_detected_object_set_bounding_box
        do_sbb.argtypes = [self.C_TYPE_PTR, BoundingBox.C_TYPE_PTR]
        return do_sbb(self, bbox)

    def confidence(self):
        do_conf = self.VITAL_LIB.vital_detected_object_confidence
        do_conf.argtypes = [self.C_TYPE_PTR]
        do_conf.restype = ctypes.c_double
        return do_sbb(self)

    def set_confidence(self, confid):
        do_sc = self.VITAL_LIB.vital_detected_object_set_confidence
        do_sc.argtypes = [self.C_TYPE_PTR, ctypes.c_double]
        do_sbb(self, confid)

    def type(self):
        do_ty = self.VITAL_LIB.vital_detected_object_get_type
        do_ty.argtypes = [self.C_TYPE_PTR]
        do_ty.restype = DetectedObjectType.C_TYPE_PTR
        return do_ty(self)

    def set_type(self, ob_type):
        do_ty = self.VITAL_LIB.vital_detected_object_set_type
        do_ty.argtypes = [self.C_TYPE_PTR, DetectedObjectType.C_TYPE_PTR]
        do_ty(self, ob_type)
