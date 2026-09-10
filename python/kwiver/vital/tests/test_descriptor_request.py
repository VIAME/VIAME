"""
ckwg +31
Copyright 2020 by Kitware, Inc.
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

Tests for vital::descriptor_request interface

"""

import numpy.testing as npt
import numpy as np
from kwiver.vital.types import (
    BoundingBoxD as bbD,
    BoundingBoxI as bbI,
    Image,
    ImageContainer,
    DescriptorRequest,
    Timestamp,
    UID,
)
import unittest


class TestVitalDescriptorRequest(unittest.TestCase):
    def test_create(self):
        DescriptorRequest()

    def test_set_and_get_id(self):
        dr = DescriptorRequest()

        # First check default
        self.assertEqual(dr.id.value(), "")
        self.assertFalse(dr.id.is_valid())

        # Now check setting and getting a few values
        dr.id = UID("first")
        self.assertEqual(dr.id.value(), "first")

        dr.id = UID("second")
        self.assertEqual(dr.id.value(), "second")

        dr.id = UID("42")
        self.assertEqual(dr.id.value(), "42")

        # Try setting back to empty
        dr.id = UID()
        self.assertEqual(dr.id.value(), "")

    def test_bad_set_id(self):
        with self.assertRaises(TypeError):
            dr = DescriptorRequest()
            dr.id = "string, not uid"

    def test_set_and_get_temporal_bounds(self):
        dr = DescriptorRequest()

        # First check the defaults
        self.assertFalse(dr.temporal_lower_bound().is_valid())
        self.assertFalse(dr.temporal_upper_bound().is_valid())

        test_bounds = [
            (Timestamp(100, 1), Timestamp(100, 1)),
            (Timestamp(100, 1), Timestamp(200, 2)),
            (Timestamp(300, 5), Timestamp(400, 6)),
        ]

        for t1, t2 in test_bounds:
            dr.set_temporal_bounds(t1, t2)
            self.assertEqual(dr.temporal_lower_bound(), t1)
            self.assertEqual(dr.temporal_upper_bound(), t2)

        dr.set_temporal_bounds(Timestamp(), Timestamp())
        self.assertFalse(dr.temporal_lower_bound().is_valid())
        self.assertFalse(dr.temporal_upper_bound().is_valid())

    def test_bad_set_temporal_bounds(self):
        with self.assertRaises(TypeError):
            dr = DescriptorRequest()
            dr.set_temporal_bounds("string", "another_string")

    def test_get_set_spatial_regions(self):
        dr = DescriptorRequest()
        ul = np.array([0, 0])
        lr = np.array([720, 1080])
        b = bbI(ul, lr)
        dr.spatial_regions = np.array([b])
        b_arr = dr.spatial_regions
        npt.assert_array_equal(b_arr[0].upper_left(), ul)

    def test_bad_set_spatial_regions(self):
        with self.assertRaises(TypeError):
            dr = DescriptorRequest()
            dr.spatial_regions = "string, not list"

    def test_set_and_get_image_data(self):
        dr = DescriptorRequest()

        imc_list = [ImageContainer(Image())]
        dr.image_data = imc_list
        self.assertEqual(len(dr.image_data), len(imc_list))
        self.assertEqual(len(imc_list), 1)
        self.assertEqual(dr.image_data[0].size(), imc_list[0].size())
        self.assertEqual(imc_list[0].size(), 0)

        imc_list.append(ImageContainer(Image(720, 480)))
        dr.image_data = imc_list
        self.assertEqual(len(dr.image_data), len(imc_list))
        self.assertEqual(len(imc_list), 2)
        self.assertEqual(dr.image_data[0].size(), imc_list[0].size())
        self.assertEqual(imc_list[0].size(), 0)
        self.assertEqual(dr.image_data[1].size(), imc_list[1].size())
        self.assertEqual(imc_list[1].size(), 720 * 480)

        dr.image_data = []
        self.assertEqual(len(dr.image_data), 0)

    def test_bad_set_image_data(self):
        with self.assertRaises(TypeError):
            dr = DescriptorRequest()
            dr.image_data = "string, not image_data"

    def test_set_and_get_data_location(self):
        dr = DescriptorRequest()

        self.assertEqual(dr.data_location, "")

        dr.data_location = "first"
        self.assertEqual(dr.data_location, "first")

        dr.data_location = "second"
        self.assertEqual(dr.data_location, "second")

        dr.data_location = "42"
        self.assertEqual(dr.data_location, "42")

        dr.data_location = ""
        self.assertEqual(dr.data_location, "")

    def test_bad_set_data_location(self):
        with self.assertRaises(TypeError):
            dr = DescriptorRequest()
            dr.data_location = 5
