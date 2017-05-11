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

Interface to VITAL image_container class.

"""
import ctypes
from vital.util import VitalObject
from vital.util import VitalErrorHandle


class DetectedObjectType (VitalObject):
    """
    vital::detected_object_type interface class
    """

    def __init__(self):
        """
        Create a simple detected object type

         """
        super(DetectedObjectType, self).__init__()

    def _new(self, count, names, scores):
        """
        Create a new type container
        """
        if (count is None or names is None or scores is None):
            dot_new = self.VITAL_LIB.vital_detected_object_type_new()
            dot_new.argtypes = []
            dot_new.restype = self.C_TYPE_PTR
            return dot_new()
        else:
            dot_nfl = self.VITAL_LIB.vital_detected_object_type_new_from_list
            dot_nfl.argtypes = [ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_char_p),
                                ctypes.POINTER(ctypes.c_double)]
            dot_nfl.restype = self.C_TYPE_PTR
            return dot_nfl(self, count, names, scores)

    def _destroy(self):
        dot_del = self.VITAL_LIB.vital_detected_object_type_destroy
        dot_del.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        with VitalErrorHandle() as eh:
            dot_del(self, eh)

    def has_class_name(self, name):
        dot_hcn = self.VITAL_LIB.vital_detected_object_type_has_class_name
        dot_hcn.argtypes = [self.C_TYPE_PTR, ctypes.c_char_p]
        dot_hcn.restype = ctypes.c_bool
        return dot_hcn(self, name)

    def score(self, name):
        dot_score = self.VITAL_LIB.vital_detected_object_type_score
        dot_score.argtypes = [self.C_TYPE_PTR, ctypes.c_char_p]
        dot_score.restype = c_double
        return dot_score(self, name )

    def get_most_likely_class(self):
        dot_gmlc = self.VITAL_LIB.vital_detected_object_type_get_most_likely_class
        dot_gmlc.argtypes = [self.C_TYPE_PTR]
        dot_gmlc.restype = c_char_p
        return dot_gmlc(self)

    def get_most_likely_score(self):
        dot_gmls = self.VITAL_LIB.vital_detected_object_type_get_most_likely_score
        dot_gmls.argtypes = [self.C_TYPE_PTR]
        dot_gmls.restype = c_double
        return dot_gmls(self)

    def set_score(self, name, score):
        dot_ss = self.VITAL_LIB.vital_detected_object_type_set_score
        dot_ss.argtypes = [self.C_TYPE_PTR, ctypes.c_char_p, ctypes.c_double]
        dot_ss(self, name, score)

    def delete_score(self, name):
        dot_ds = self.VITAL_LIB.vital_detected_object_type_delete_score
        dot_ds.argtypes[self.C_TYPE_PTR, ctypes.c_char_p]
        dot_ds(self, name)

    def class_names(self):
        dot_cn = self.VITAL_LIB.vital_detected_object_type_class_names
        dot_cn.argtypes = [self.C_TYPE_PTR]
        dot_cn.restype = ctypes.POINTER(ctypes.c_char_p)
        return dot_cn(self)

    def all_class_names(self):
        dot_acn = self.VITAL_LIB.vital_detected_object_type_all_class_names
        dot_acn.argtypes = [self.C_TYPE_PTR]
        dot_acn.restype = ctypes.POINTER(ctypes.c_char_p)
        return dot_acn(self)
