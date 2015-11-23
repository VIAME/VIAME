"""
ckwg +31
Copyright 2015 by Kitware, Inc.
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

Functions to convert sprokit pipeline datum to vital types.

These python functions provide an interface to the underlying "C"
implementation of type converters.

"""
# -*- coding: utf-8 -*-

import ctypes

from vital import ImageContainer
from vital import TrackSet
from vital.util import find_library_path

__VITAL_CONVERTERS_LIB__ = None

def _find_converter_lib():
    #
    # Load our supporting library
    # TBD call     VITAL_LIB = find_vital_library()
    # or similar to locate library.
    global __VITAL_CONVERTERS_LIB__
    if not __VITAL_CONVERTERS_LIB__:
        lib_path = find_library_path("vital_type_converters")
        if not lib_path:
            raise RuntimeError( "Unable to locate 'vital_type_converters' support library")

        __VITAL_CONVERTERS_LIB__ = ctypes.CDLL(lib_path)
        if not __VITAL_CONVERTERS_LIB__:
            raise RuntimeError("Unable to locate vital_type_converters")

    return __VITAL_CONVERTERS_LIB__



def _convert_image_container_sptr(datum_ptr):
    """
    Convert datum to image_container opaque handle.
    """
    _VCL = _find_converter_lib()
    func = _VCL.vital_image_container_from_datum
    func.argtypes = [ ctypes.py_object ]
    func.restype = ImageContainer.C_TYPE_PTR
    return func(datum_ptr)


def _convert_track_set_sptr(datum_ptr):
    """
    Convert datum to track set.
    """
    func = _find_converter_lib().vital_trackset_from_datum
    func.argtypes = [ ctypes.py_object ]
    func.restype = TrackSet.C_TYPE_PTR
    return func(datum_ptr)




"""
Converters to do:
    feature_set
    descriptor_set


"""
