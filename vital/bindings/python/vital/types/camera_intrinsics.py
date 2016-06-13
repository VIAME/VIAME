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

Interface to VITAL camera_intrinsics objects

"""
import collections
import ctypes

import numpy

from vital.types.eigen import EigenArray
from vital.util import VitalErrorHandle, VitalObject


class CameraIntrinsics (VitalObject):

    def __init__(self, focal_length=1., principle_point=(0, 0),
                 aspect_ratio=1., skew=0., dist_coeffs=(), from_cptr=None):
        """
        :param focal_length: Focal length (default=1.0)
        :type focal_length: float

        :param principle_point:  Principle point (default: [0,0]).
            Values are copied into this structure.
        :type principle_point: collections.Sequence[float]

        :param aspect_ratio: Aspect ratio (default: 1.0)
        :type aspect_ratio: float

        :param skew: Skew (default: 0.0)
        :type skew: float

        :param dist_coeffs: Existing distortion coefficients (Default: empty).
            Values are copied into this structure.
        :type dist_coeffs: collections.Sequence[float]
        """
        super(CameraIntrinsics, self).__init__(from_cptr, focal_length,
                                               principle_point, aspect_ratio,
                                               skew, dist_coeffs)

    def _new(self, focal_length, principle_point, aspect_ratio, skew,
             dist_coeffs):
        """
        Construct a new vital::camera_intrinsics instance
        :type focal_length: float
        :type principle_point: collections.Sequence[float]
        :type aspect_ratio: float
        :type skew: float
        :type dist_coeffs: collections.Sequence[float]
        """
        ci_new = self.VITAL_LIB['vital_camera_intrinsics_new']
        ci_new.argtypes = [
            ctypes.c_double,
            EigenArray.c_ptr_type(2, 1, ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            EigenArray.c_ptr_type('X', 1, ctypes.c_double),
            VitalErrorHandle.C_TYPE_PTR,
        ]
        ci_new.restype = self.C_TYPE_PTR
        # Make "vectors"
        pp = EigenArray(2)
        pp.T[:] = principle_point
        dc = EigenArray(len(dist_coeffs), dynamic_rows=True)
        if len(dist_coeffs):
            dc.T[:] = dist_coeffs

        with VitalErrorHandle() as eh:
            return ci_new(focal_length, pp, aspect_ratio, skew, dc,
                                    eh)

    def _destroy(self):
        ci_dtor = self.VITAL_LIB['vital_camera_intrinsics_destroy']
        ci_dtor.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        with VitalErrorHandle() as eh:
            ci_dtor(self, eh)

    @property
    def focal_length(self):
        f = self.VITAL_LIB['vital_camera_intrinsics_get_focal_length']
        f.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        f.restype = ctypes.c_double
        with VitalErrorHandle() as eh:
            return f(self, eh)

    @property
    def principle_point(self):
        f = self.VITAL_LIB['vital_camera_intrinsics_get_principle_point']
        f.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        f.restype = EigenArray.c_ptr_type(2, 1, ctypes.c_double)
        with VitalErrorHandle() as eh:
            m_ptr = f(self, eh)
            return EigenArray(2, from_cptr=m_ptr, owns_data=True)

    @property
    def aspect_ratio(self):
        f = self.VITAL_LIB['vital_camera_intrinsics_get_aspect_ratio']
        f.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        f.restype = ctypes.c_double
        with VitalErrorHandle() as eh:
            return f(self, eh)

    @property
    def skew(self):
        f = self.VITAL_LIB['vital_camera_intrinsics_get_skew']
        f.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        f.restype = ctypes.c_double
        with VitalErrorHandle() as eh:
            return f(self, eh)

    @property
    def dist_coeffs(self):
        """ Get the distortion coefficients array """
        f = self.VITAL_LIB['vital_camera_intrinsics_get_dist_coeffs']
        f.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        f.restype = EigenArray.c_ptr_type('X', 1, ctypes.c_double)
        with VitalErrorHandle() as eh:
            m_ptr = f(self, eh)
            return EigenArray(dynamic_rows=1, from_cptr=m_ptr, owns_data=True)

    def __eq__(self, other):
        if isinstance(other, CameraIntrinsics):
            return (
                self.focal_length == other.focal_length and
                numpy.allclose(self.principle_point, other.principle_point) and
                self.aspect_ratio == other.aspect_ratio and
                self.skew == other.skew and
                numpy.allclose(self.dist_coeffs, other.dist_coeffs)
            )
        return False

    def __ne__(self, other):
        return not (self == other)

    def as_matrix(self):
        """
        Access the intrinsics as an upper triangular matrix

        **Note:** *This matrix includes the focal length, principal point,
        aspect ratio, and skew, but does not model distortion.*

        :return: 3x3 upper triangular matrix

        """
        f = self.VITAL_LIB['vital_camera_intrinsics_as_matrix']
        f.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        f.restype = EigenArray.c_ptr_type(3, 3, ctypes.c_double)
        with VitalErrorHandle() as eh:
            m_ptr = f(self, eh)
            return EigenArray(3, 3, from_cptr=m_ptr, owns_data=True)

    def map_2d(self, norm_pt):
        """
        Map normalized image coordinates into actual image coordinates

        This function applies both distortion and application of the
        calibration matrix to map into actual image coordinates.

        :param norm_pt: Normalized image coordinate to map to an image
            coordinate (2-element sequence).
        :type norm_pt: collections.Sequence[float]

        :return: Mapped 2D image coordinate
        :rtype: EigenArray[float]

        """
        assert len(norm_pt) == 2, "Input sequence was not of length 2"
        f = self.VITAL_LIB['vital_camera_intrinsics_map_2d']
        f.argtypes = [self.C_TYPE_PTR,
                      EigenArray.c_ptr_type(2, 1, ctypes.c_double),
                      VitalErrorHandle.C_TYPE_PTR]
        f.restype = EigenArray.c_ptr_type(2, 1, ctypes.c_double)
        p = EigenArray(2)
        p.T[:] = norm_pt
        with VitalErrorHandle() as eh:
            m_ptr = f(self, p, eh)
            return EigenArray(2, 1, from_cptr=m_ptr, owns_data=True)

    def map_3d(self, norm_hpt):
        """
        Map a 3D point in camera coordinates into actual image coordinates

        :param norm_hpt: Normalized coordinate to map to an image coordinate
            (3-element sequence)
        :type norm_hpt: collections.Sequence[float]

        :return: Mapped 2D image coordinate
        :rtype: EigenArray[float]

        """
        assert len(norm_hpt) == 3, "Input sequence was not of length 3"
        f = self.VITAL_LIB['vital_camera_intrinsics_map_3d']
        f.argtypes = [self.C_TYPE_PTR,
                      EigenArray.c_ptr_type(3, 1, ctypes.c_double),
                      VitalErrorHandle.C_TYPE_PTR]
        f.restype = EigenArray.c_ptr_type(2, 1, ctypes.c_double)
        p = EigenArray(3)
        p.T[:] = norm_hpt
        with VitalErrorHandle() as eh:
            m_ptr = f(self, p, eh)
            return EigenArray(2, 1, from_cptr=m_ptr, owns_data=True)

    def unmap_2d(self, pt):
        """
        Unmap actual image coordinates back into normalized image coordinates

        This function applies both application of the inverse calibration matrix
        and undistortion of the normalized coordinates

        :param pt: Actual image 2D point to un-map.

        :return: Un-mapped normalized image coordinate.

        """
        assert len(pt) == 2, "Input sequence was not of length 2"
        f = self.VITAL_LIB['vital_camera_intrinsics_unmap_2d']
        f.argtypes = [self.C_TYPE_PTR,
                      EigenArray.c_ptr_type(2, 1, ctypes.c_double),
                      VitalErrorHandle.C_TYPE_PTR]
        f.restype = EigenArray.c_ptr_type(2, 1, ctypes.c_double)
        p = EigenArray(2)
        p.T[:] = pt
        with VitalErrorHandle() as eh:
            m_ptr = f(self, p, eh)
            return EigenArray(2, 1, from_cptr=m_ptr, owns_data=True)

    def distort_2d(self, norm_pt):
        """
        Map normalized image coordinates into distorted coordinates

        :param norm_pt: Normalized 2D image coordinate.

        :return: Distorted 2D coordinate.

        """
        assert len(norm_pt) == 2, "Input sequence was not of length 2"
        f = self.VITAL_LIB['vital_camera_intrinsics_distort_2d']
        f.argtypes = [self.C_TYPE_PTR,
                      EigenArray.c_ptr_type(2, 1, ctypes.c_double),
                      VitalErrorHandle.C_TYPE_PTR]
        f.restype = EigenArray.c_ptr_type(2, 1, ctypes.c_double)
        p = EigenArray(2)
        p.T[:] = norm_pt
        with VitalErrorHandle() as eh:
            m_ptr = f(self, p, eh)
            return EigenArray(2, 1, from_cptr=m_ptr, owns_data=True)

    def undistort_2d(self, dist_pt):
        """
        Unmap distorted normalized coordinates into normalized coordinates

        :param dist_pt: Distorted 2D coordinate to un-distort.

        :return: Normalized 2D image coordinate.

        """
        assert len(dist_pt) == 2, "Input sequence was not of length 2"
        f = self.VITAL_LIB['vital_camera_intrinsics_undistort_2d']
        f.argtypes = [self.C_TYPE_PTR,
                      EigenArray.c_ptr_type(2, 1, ctypes.c_double),
                      VitalErrorHandle.C_TYPE_PTR]
        f.restype = EigenArray.c_ptr_type(2, 1, ctypes.c_double)
        p = EigenArray(2)
        p.T[:] = dist_pt
        with VitalErrorHandle() as eh:
            m_ptr = f(self, p, eh)
            return EigenArray(2, 1, from_cptr=m_ptr, owns_data=True)
