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

Interface to the VITAL descriptor_set class.

"""
import ctypes

from vital.types import Descriptor
from vital.util import VitalObject, free_void_ptr


class DescriptorSet (VitalObject):
    """
    vital::descriptor_set interface class
    """

    def __init__(self, descriptor_list=None, from_cptr=None):
        super(DescriptorSet, self).__init__(from_cptr, descriptor_list)

    def _new(self, descriptor_list):
        """
        Create a new descriptor set instance from a list of descriptor elements.

        :param descriptor_list: List of child descriptor instance handles.
        :type descriptor_list: list[Descriptor]

        """
        if descriptor_list is None:
            descriptor_list = []
        c_descriptor_array = (Descriptor.c_ptr_type() * len(descriptor_list))(
            *(d.c_pointer for d in descriptor_list)
        )
        return self._call_cfunc(
            'vital_descriptor_set_new',
            [ctypes.POINTER(Descriptor.c_ptr_type()), ctypes.c_size_t],
            [c_descriptor_array, len(descriptor_list)],
            self.C_TYPE_PTR
        )

    def _destroy(self):
        """
        Destroy our descriptor set handle.
        """
        self._call_cfunc(
            'vital_descriptor_set_destroy',
            [self.C_TYPE_PTR],
            [self],
        )

    def size(self):
        """
        :return: Number of descriptors contained in this descriptor set.
        :rtype: long
        """
        return self._call_cfunc(
            'vital_descriptor_set_size',
            [self.C_TYPE_PTR],
            [self],
            ctypes.c_size_t
        )

    def __len__(self):
        return self.size()

    def descriptors(self):
        """
        Get the set of descriptor instances contained in this set as a tuple.

        :return: A tuple of the contained descriptor instances.
        :rtype: tuple[Descriptor]
        """
        out_d_array = ctypes.POINTER(Descriptor.c_ptr_type())()
        out_d_array_size = ctypes.c_size_t()

        self._call_cfunc(
            'vital_descriptor_set_get_descriptors',
            [self.C_TYPE_PTR,
             ctypes.POINTER(ctypes.POINTER(Descriptor.c_ptr_type())),
             ctypes.POINTER(ctypes.c_size_t)],
            [self, ctypes.byref(out_d_array), ctypes.byref(out_d_array_size)],
        )

        d_list = []
        for i in range(out_d_array_size.value):
            cptr = Descriptor.c_ptr_type()(out_d_array[i].contents)
            d_list.append(Descriptor(from_cptr = cptr))
        free_void_ptr(out_d_array)

        return d_list
