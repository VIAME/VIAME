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

Interface to VITAL detected_object_set class.

"""
import ctypes

from vital.types import DetectedObject
from vital.util import free_void_ptr
from vital.util import VitalObject
from vital.util import VitalErrorHandle


class DetectedObjectSet (VitalObject):
    """
    vital::detected_object_set interface class
    """

    def __init__(self, dobjs = None, count = None, from_cptr = None):
        """
        Create a simple detected object type

         """
        super(DetectedObjectSet, self).__init__(from_cptr, dobjs, count)

    def _new(self, dobjs = None, count = None):
        if dobjs is None or count is None:
            dos_new = self.VITAL_LIB.vital_detected_object_set_new
            dos_new.argtypes = []
            dos_new.restype = self.C_TYPE_PTR
            return dos_new()
        else:
            dos_nfl = self.VITAL_LIB.vital_detected_object_set_new_from_list
            dos_nfl.argtypes = [ctypes.POINTER( DetectedObject.C_TYPE_PTR), ctypes.c_size_t]
            dos_nfl.restype = self.C_TYPE_PTR
            return dos_nfl(dobjs, count)

    def _destroy(self):
        dos_del = self.VITAL_LIB.vital_detected_object_set_destroy
        dos_del.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        with VitalErrorHandle() as eh:
            dos_del(self, eh)

    def add(self, dobj):
        dos_add = self.VITAL_LIB.vital_detected_object_set_add
        dos_add.argtypes = [self.C_TYPE_PTR, DetectedObject.C_TYPE_PTR]
        dos_add(self, dobj)

    def size(self):
        dos_size = self.VITAL_LIB.vital_detected_object_set_size
        dos_size.argtypes = [self.C_TYPE_PTR]
        dos_size.restype = ctypes.c_size_t
        return dos_size(self)

    def select(self, one = 0.0, two = None):
        
        length = ctypes.c_size_t() 

        if two is None:
            dos_st = self.VITAL_LIB.vital_detected_object_set_select_threshold
            dos_st.argtypes = [self.C_TYPE_PTR, ctypes.c_double, ctypes.POINTER(ctypes.c_size_t)]
            dos_st.restype = ctypes.POINTER( DetectedObject.c_ptr_type() )
            c_output = dos_st( self, one, ctypes.byref(length) )
        else:
            dos_sct = self.VITAL_LIB.vital_detected_object_set_select_class_threshold
            dos_sct.argtypes = [self.C_TYPE_PTR, ctypes.c_char_p, ctypes.c_double, ctypes.POINTER(ctypes.c_size_t)]
            dos_sct.restype = ctypes.POINTER( DetectedObject.c_ptr_type() )
            c_output = dos_sct(self, one, two, ctypes.byref(length) )

        output = []
        for i in range( length.value ):
            do_new = self.VITAL_LIB.vital_detected_object_copy
            do_new.argtypes = [DetectedObject.C_TYPE_PTR]
            do_new.restype = DetectedObject.C_TYPE_PTR
            output.append( DetectedObject( from_cptr=do_new(c_output[i]) ) )

        free_void_ptr( c_output )
        return output

