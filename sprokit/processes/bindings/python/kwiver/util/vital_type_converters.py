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

from vital.types import ImageContainer
from vital.types import TrackSet
from vital.types import DetectedObjectSet
from vital.util import find_vital_library

__VITAL_CONVERTERS_LIB__ = None

def _find_converter_lib():
    #
    # Load our supporting library
    #
    # We are caching the lib interface here.
    global __VITAL_CONVERTERS_LIB__
    if not __VITAL_CONVERTERS_LIB__:
        lib_path = find_vital_library.find_vital_type_converter_library_path()

        if not lib_path:
            raise RuntimeError( "Unable to locate 'vital_type_converters' support library")

        __VITAL_CONVERTERS_LIB__ = ctypes.CDLL(lib_path)
        if not __VITAL_CONVERTERS_LIB__:
            raise RuntimeError("Unable to locate vital_type_converters")

    return __VITAL_CONVERTERS_LIB__


def _convert_image_container_in(datum_ptr):
    """
    Convert datum as PyCapsule to image_container opaque handle.
    """
    _VCL = _find_converter_lib()
    # Convert from datum to opaque handle.
    func = _VCL.vital_image_container_from_datum
    func.argtypes = [ ctypes.py_object ]
    func.restype = ImageContainer.C_TYPE_PTR
    # get opaque handle from the datum
    handle = func(datum_ptr)

    # convert handle to python object - from c-ptr
    py_ic_obj = ImageContainer( None, handle )

    return py_ic_obj


def _convert_image_container_out(handle):
    """
    Convert datum as PyCapsule from image_container opaque handle.
    """
    _VCL = _find_converter_lib()
    # convert opaque handle to datum (as PyCapsule)
    func =  _VCL.vital_image_container_to_datum
    func.argtypes = [ ImageContainer.C_TYPE_PTR ]
    func.restype = ctypes.py_object
    retval = func(handle)
    return retval


def _convert_detected_object_set_in(datum_ptr):
    """
    Convert datum as PyCapsule to image_container opaque handle.
    """
    _VCL = _find_converter_lib()
    # Convert from datum to opaque handle.
    func = _VCL.vital_detected_object_set_from_datum
    func.argtypes = [ ctypes.py_object ]
    func.restype = DetectedObjectSet.C_TYPE_PTR
    # get opaque handle from the datum
    handle = func(datum_ptr)

    # convert handle to python object - from c-ptr
    py_ic_obj = DetectedObjectSet( None, handle )

    return py_ic_obj


def _convert_detected_object_set_out(handle):
    """
    Convert datum as PyCapsule from image_container opaque handle.
    """
    _VCL = _find_converter_lib()
    # convert opaque handle to datum (as PyCapsule)
    func =  _VCL.vital_detected_object_set_to_datum
    func.argtypes = [ DetectedObjectSet.C_TYPE_PTR ]
    func.restype = ctypes.py_object
    retval = func(handle)
    return retval


# ------------------------------------------------------------------
def _convert_double_vector_in( datum_ptr ):
    """
    Convert datum pointer to python list.
    """
    _VCL = _find_converter_lib()
    func =  _VCL.double_vector_from_datum
    func.argtypes = [ ctypes.py_object ]
    func.restype = ctypes.POINTER(ctypes.c_double) # may need a tuple here to return length
    return func( datum_ptr )


def _convert_double_vector_out( dlist ):
    """
    Convert python list to datum as PyCapsule.

    Possibly check type of input and handle arrays amd nparrays too.
    Convert to standard form for C translation to datum.
    """
    _VCL = _find_converter_lib()
    func =  _VCL.double_vector_to_datum
    func.argtypes = [ ctypes.py_object ]
    func.restype = ctypes.py_object
    return func( dlist )


# ------------------------------------------------------------------
def _convert_track_set_handle(datum_ptr):
    """
    Convert datum to track set.

    Note: not tested so code is surely wrong. See above for
    template code.
    """
    func = _find_converter_lib().vital_trackset_from_datum
    func.argtypes = [ ctypes.py_object ]
    func.restype = TrackSet.C_TYPE_PTR
    return func(datum_ptr)




"""
Converters to do:
    feature_set
    descriptor_set
    detected object classes

"""
