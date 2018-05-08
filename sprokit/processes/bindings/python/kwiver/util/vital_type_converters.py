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

from vital.types import DescriptorSet
from vital.types import DetectedObjectSet
from vital.types import ImageContainer
from vital.types import TrackSet
from vital.util import find_vital_library
from vital.util.string import vital_string_t


def _convert_image_container_in(datum_ptr):
    """
    Convert datum as PyCapsule to image_container opaque handle.
    """
    _VCL = find_vital_library.find_vital_type_converter_library()
    # Convert from datum to opaque handle.
    func = _VCL.vital_image_container_from_datum
    func.argtypes = [ctypes.py_object]
    func.restype = ImageContainer.C_TYPE_PTR
    # get opaque handle from the datum
    handle = func(datum_ptr)

    # convert handle to python object - from c-ptr
    py_ic_obj = ImageContainer(None, handle)

    return py_ic_obj


def _convert_image_container_out(handle):
    """
    Convert datum as PyCapsule from image_container opaque handle.
    """
    _VCL = find_vital_library.find_vital_type_converter_library()
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
    _VCL = find_vital_library.find_vital_type_converter_library()
    # Convert from datum to opaque handle.
    func = _VCL.vital_detected_object_set_from_datum
    func.argtypes = [ ctypes.py_object ]
    func.restype = DetectedObjectSet.C_TYPE_PTR
    # get opaque handle from the datum
    handle = func(datum_ptr)

    # convert handle to python object - from c-ptr
    return DetectedObjectSet( from_cptr=handle )


def _convert_detected_object_set_out(handle):
    """
    Convert datum as PyCapsule from image_container opaque handle.
    """
    _VCL = find_vital_library.find_vital_type_converter_library()
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
    _VCL = find_vital_library.find_vital_type_converter_library()
    func = _VCL.double_vector_from_datum
    func.argtypes = [ctypes.py_object]
    # may need a tuple here to return length
    func.restype = ctypes.POINTER(ctypes.c_double)
    return func(datum_ptr)


def _convert_double_vector_out(dlist):
    """
    Convert python list to datum as PyCapsule.

    Possibly check type of input and handle arrays amd nparrays too.
    Convert to standard form for C translation to datum.
    """
    _VCL = find_vital_library.find_vital_type_converter_library()
    func = _VCL.double_vector_to_datum
    func.argtypes = [ctypes.py_object]
    func.restype = ctypes.py_object
    return func(dlist)


# ------------------------------------------------------------------
def _convert_track_set_in(datum_ptr):
    """
    Convert datum to track set.
    """
    _VCL = find_vital_library.find_vital_type_converter_library()
    func = _VCL.vital_trackset_from_datum
    func.argtypes = [ ctypes.py_object ]
    func.restype = TrackSet.C_TYPE_PTR
    return func(datum_ptr)


# ------------------------------------------------------------------
def _convert_track_set_out(handle):
    """
    Convert track set to datum
    """
    _VCL = find_vital_library.find_vital_type_converter_library()
    func = _VCL.vital_trackset_to_datum
    func.argtypes = [ TrackSet.C_TYPE_PTR ]
    func.restype = ctypes.py_object
    return func(handle)


# ------------------------------------------------------------------
def _convert_object_track_set_in(datum_ptr):
    """
    Convert datum to track set.
    """
    _VCL = find_vital_library.find_vital_type_converter_library()
    func = _VCL.vital_object_trackset_from_datum
    func.argtypes = [ ctypes.py_object ]
    func.restype = TrackSet.C_TYPE_PTR
    return func(datum_ptr)


# ------------------------------------------------------------------
def _convert_object_track_set_out(handle):
    """
    Convert track set to datum
    """
    _VCL = find_vital_library.find_vital_type_converter_library()
    func = _VCL.vital_object_trackset_to_datum
    func.argtypes = [ TrackSet.C_TYPE_PTR ]
    func.restype = ctypes.py_object
    return func(handle)
    

# ------------------------------------------------------------------
def convert_string_vector_in(datum_ptr):
    """
    Convert a datum pointer into a python tuple of strings.

    :param datum_ptr: Sprokit datum pointer.

    :return: List of strings
    :rtype: list[str]

    """
    _VL = find_vital_library.find_vital_library()
    _VCL = find_vital_library.find_vital_type_converter_library()

    func = _VCL['vital_string_vector_from_datum']
    func.argtypes = [ctypes.py_object,
                     ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
                     ctypes.POINTER(ctypes.c_size_t)]
    sl_free = _VL['vital_common_free_string_list']
    sl_free.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p)]

    c_arr_strings = ctypes.POINTER(ctypes.c_char_p)()
    c_arr_size = ctypes.c_size_t()
    func(datum_ptr, ctypes.byref(c_arr_strings), ctypes.byref(c_arr_size))

    # Convert output array into tuple of strings
    s_list = []
    for i in range(c_arr_size.value):
        s_list.append(c_arr_strings[i])

    # Free strings allocated in C function.
    sl_free(c_arr_size, c_arr_strings)

    return s_list


def convert_string_vector_out(py_strings):
    """
    Convert an iterable of python strings to a sprokit datum.

    :param py_strings: Some iterable of strings.
    :type py_strings: collections.Iterable[str]

    :return: Datum pointer.

    """
    _VCL = find_vital_library.find_vital_type_converter_library()

    func = _VCL['vital_string_vector_to_datum']
    func.argtypes = [ctypes.py_object]
    func.restype = ctypes.py_object

    s_list = list(py_strings)
    # Return generated capsule PyObject
    return func(s_list)


# ------------------------------------------------------------------
def convert_descriptor_set_in(datum_ptr):
    """
    Convert a datum pointer to a vital DescriptorSet instance.

    :param datum_ptr: Sprokit datum pointer

    :return: Vital DescriptorSet instance
    :rtype: DescriptorSet

    """
    _VCL = find_vital_library.find_vital_type_converter_library()

    func = _VCL['vital_descriptor_set_from_datum']
    func.argtypes = [ctypes.py_object]
    func.restype = DescriptorSet.c_ptr_type()

    ds_handle = func(datum_ptr)
    return DescriptorSet(from_cptr=ds_handle)


def convert_descriptor_set_out(py_descriptor_set):
    """
    Convert a vital python DescriptorSet instance to a datum as a PyCapsule.

    :param py_descriptor_set: The vital DescriptorSet instance to convert.
    :type py_descriptor_set: DescriptorSet

    :return: Datum pointer.

    """
    _VCL = find_vital_library.find_vital_type_converter_library()

    func = _VCL['vital_descriptor_set_to_datum']
    func.argtypes = [DescriptorSet.c_ptr_type()]
    func.restype = ctypes.py_object

    return func(py_descriptor_set)

"""
Converters to do:
    feature_set
    detected object classes
"""
