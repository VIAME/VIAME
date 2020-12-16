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

vital::descriptor_set interface tests

"""
import unittest

import numpy

from kwiver.vital.types import new_descriptor, DescriptorSet


class TestDescriptorSet (unittest.TestCase):

    def test_new_empty(self):
        # Create an empty descriptor set, checking that no errors occur.
        DescriptorSet()

    def test_new_with_descriptors(self):
        # Try creating a descriptor set with multiple descriptors as input.
        descriptor_list = [
            new_descriptor(1),
            new_descriptor(1),
            new_descriptor(1),
        ]
        ds = DescriptorSet(descriptor_list)

    def test_size_empty(self):
        # Check size of descriptor set created with no descriptor list.
        ds = DescriptorSet()
        self.assertEqual(ds.size(), 0)

        # len() should also function and return the same thing
        self.assertEqual(len(ds), 0)

    def test_size_multiple(self):
        # Check that size accurately report number of descriptors constructed
        # with.
        d_list = [
            new_descriptor(),
        ]
        ds = DescriptorSet(d_list)
        self.assertEqual(ds.size(), 1)
        self.assertEqual(len(ds), 1)

    def test_get_descriptors_empty(self):
        # Try getting descriptor list from a set constructed with no
        # descriptors.
        ds = DescriptorSet()
        r_list = ds.descriptors()
        self.assertEqual(len(r_list), 0)

    def test_get_descriptors_multiple(self):
        # Test getting descriptors given to the set in its constructor.
        d_list = [
            new_descriptor(1),
            new_descriptor(2),
            new_descriptor(3),
        ]
        d_list[0][:] = [0]
        d_list[1][:] = [1, 2]
        d_list[2][:] = [3, 4, 5]

        ds = DescriptorSet(d_list)

        ds_descriptors = ds.descriptors()
        self.assertEqual(len(ds), 3)
        self.assertEqual(len(ds_descriptors), 3)
        numpy.testing.assert_array_equal(ds_descriptors[0], d_list[0])
        numpy.testing.assert_array_equal(ds_descriptors[1], d_list[1])
        numpy.testing.assert_array_equal(ds_descriptors[2], d_list[2])
