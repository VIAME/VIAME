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

Interface to VITAL bounding_box class.

"""
import ctypes

from vital.util import VitalObject
from vital.util import VitalErrorHandle


class BoundingBox (VitalObject):
    """
    vital::detected_object_ interface class
    """

    def __init__(self, one = None, two = None, three = None, four = None, from_cptr=None):
        """
        Create a simple detected object type

         """
        if from_cptr is not None:
            bb_nfb = self.VITAL_LIB.vital_bounding_box_new_from_box
            bb_nfb.argtypes = [self.C_TYPE_PTR]
            bb_nfb.restype = self.C_TYPE_PTR
            VitalObject._inst_ptr = bb_nfb( from_cptr )
        else:
            super(BoundingBox, self).__init__( from_cptr, one, two, three, four )

    def _new(self, one, two, three, four):
        """
        Has several signatures:
        1 -(double* ul, double* lr)
        2 -(double* ul, double  width, double height)
        3- (double xmin, double  ymin, double xmax, double ymax)
        """
        if (one is None or two is None):
            raise ValueError("Need at least two parameters")

        if three is None:
            """
            signature 1
            """
            bb_nfv = self.VITAL_LIB.vital_bounding_box_new_from_vectors
            bb_nfv.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            bb_nfv.restype = self.C_TYPE_PTR
            pt1 = (ctypes.c_double * 2)( *one )
            pt2 = (ctypes.c_double * 2)( *two )
            return bb_nfv( pt1, pt2 )

        elif four is None:
            """
            signature 2
            """
            bb_npwh = self.VITAL_LIB.vital_bounding_box_new_from_point_width_height
            bb_npwh.argtypes = [ctypes.POINTER(ctypes.c_double),
                                ctypes.c_double,
                                ctypes.c_double]
            bb_npwh.restype = self.C_TYPE_PTR
            pt1 = (ctypes.c_double * 2)( *one )
            return bb_npwh( pt1, ctypes.c_double( two ), ctypes.c_double( three ) )

        else:
            """
            signature 3
            """
            bb_nfc = self.VITAL_LIB.vital_bounding_box_new_from_coordinates
            bb_nfc.argtypes = [ctypes.c_double, ctypes.c_double,
                               ctypes.c_double, ctypes.c_double]
            bb_nfc.restype = self.C_TYPE_PTR
            return bb_nfc( ctypes.c_double( one ),
                           ctypes.c_double( two ),
                           ctypes.c_double( three ),
                           ctypes.c_double( four ) )

    def _destroy(self):
        bb_del = self.VITAL_LIB.vital_bounding_box_destroy
        bb_del.argtypes = [self.C_TYPE_PTR]
        bb_del( self )

    def center(self):
        """
        returns two doubles
        """
        bb_center = self.VITAL_LIB.vital_bounding_box_center
        bb_center.argtypes = [self.C_TYPE_PTR]
        bb_center.restype = ctypes.POINTER(ctypes.c_double)
        bb_center(self)

    def upper_left(self):
        """
        returns two doubles
        """
        bb_ul = self.VITAL_LIB.vital_bounding_box_upper_left
        bb_ul.argtypes = [self.C_TYPE_PTR]
        bb_ul.restype = ctypes.POINTER(ctypes.c_double)
        bb_ul(self)

    def lower_right(self):
        """
        returns two doubles
        """
        bb_lr = self.VITAL_LIB.vital_bounding_box_lower_right
        bb_lr.argtypes = [self.C_TYPE_PTR]
        bb_lr.restype = ctypes.POINTER(ctypes.c_double)
        bb_lr(self)

    def min_x(self):
        bb_minx = self.VITAL_LIB.vital_bounding_box_min_x
        bb_minx.argtypes = [self.C_TYPE_PTR]
        bb_minx.restype = ctypes.c_double
        return bb_minx(self)

    def max_x(self):
        bb_maxx = self.VITAL_LIB.vital_bounding_box_max_x
        bb_maxx.argtypes = [self.C_TYPE_PTR]
        bb_maxx.restype = ctypes.c_double
        return bb_maxx(self)

    def min_y(self):
        bb_miny = self.VITAL_LIB.vital_bounding_box_min_y
        bb_miny.argtypes = [self.C_TYPE_PTR]
        bb_miny.restype = ctypes.c_double
        return bb_miny(self)

    def max_y(self):
        bb_maxy = self.VITAL_LIB.vital_bounding_box_max_y
        bb_maxy.argtypes = [self.C_TYPE_PTR]
        bb_maxy.restype = ctypes.c_double
        return bb_maxy(self)

    def width(self):
        bb_width = self.VITAL_LIB.vital_bounding_box_width
        bb_maxy.argtypes = [self.C_TYPE_PTR]
        bb_width.restype = ctypes.c_double
        return bb_width(self)

    def height(self):
        bb_height = self.VITAL_LIB.vital_bounding_box_height
        bb_height.argtypes = [self.C_TYPE_PTR]
        bb_height.restype = ctypes.c_double
        return bb_height(self)

    def area(self):
        bb_area = self.VITAL_LIB.vital_bounding_box_area
        bb_area.argtypes = [self.C_TYPE_PTR]
        bb_area.restype = ctypes.c_double
        return bb_area(self)
