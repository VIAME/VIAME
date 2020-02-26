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
import nose.tools as nt
import numpy.testing as npt

from kwiver.vital.types import (
    BoundingBoxD as BoundingBox,
    Image,
    ImageContainer,
    DescriptorRequest,
    Timestamp,
    UID,
)


class TestVitalDescriptorRequest(object):
    def test_create(self):
        DescriptorRequest()

    def test_set_and_get_id(self):
        dr = DescriptorRequest()

        # First check default
        nt.assert_equals(dr.id.value(), "")
        nt.assert_false(dr.id.is_valid())

        # Now check setting and getting a few values
        dr.id = UID("first")
        nt.assert_equals(dr.id.value(), "first")

        dr.id = UID("second")
        nt.assert_equals(dr.id.value(), "second")

        dr.id = UID("42")
        nt.assert_equals(dr.id.value(), "42")

        # Try setting back to empty
        dr.id = UID()
        nt.assert_equals(dr.id.value(), "")

    @nt.raises(TypeError)
    def test_bad_set_id(self):
        dr = DescriptorRequest()
        dr.id = "string, not uid"

    def test_set_and_get_temporal_bounds(self):
        dr = DescriptorRequest()

        # First check the defaults
        nt.assert_false(dr.temporal_lower_bound().is_valid())
        nt.assert_false(dr.temporal_upper_bound().is_valid())

        test_bounds = [
            (Timestamp(100, 1), Timestamp(100, 1)),
            (Timestamp(100, 1), Timestamp(200, 2)),
            (Timestamp(300, 5), Timestamp(400, 6)),
        ]

        for (t1, t2) in test_bounds:
            dr.set_temporal_bounds(t1, t2)
            nt.assert_equals(dr.temporal_lower_bound(), t1)
            nt.assert_equals(dr.temporal_upper_bound(), t2)

        dr.set_temporal_bounds(Timestamp(), Timestamp())
        nt.assert_false(dr.temporal_lower_bound().is_valid())
        nt.assert_false(dr.temporal_upper_bound().is_valid())

    @nt.raises(TypeError)
    def test_bad_set_temporal_bounds(self):
        dr = DescriptorRequest()
        dr.set_temporal_bounds("string", "another_string")

    # TODO: When the PR with bindings for boundingbox<int> is merged in,
    # add test for getting and setting valid spatial regions

    @nt.raises(TypeError)
    def test_bad_set_spatial_regions(self):
        dr = DescriptorRequest()
        dr.spatial_regions = "string, not list"

    def test_set_and_get_image_data(self):
        dr = DescriptorRequest()

        imc_list = [ImageContainer(Image())]
        dr.image_data = imc_list
        nt.assert_equals(len(dr.image_data), len(imc_list))
        nt.assert_equals(len(imc_list), 1)
        nt.assert_equals(dr.image_data[0].size(), imc_list[0].size())
        nt.assert_equals(imc_list[0].size(), 0)

        imc_list.append(ImageContainer(Image(720, 480)))
        dr.image_data = imc_list
        nt.assert_equals(len(dr.image_data), len(imc_list))
        nt.assert_equals(len(imc_list), 2)
        nt.assert_equals(dr.image_data[0].size(), imc_list[0].size())
        nt.assert_equals(imc_list[0].size(), 0)
        nt.assert_equals(dr.image_data[1].size(), imc_list[1].size())
        nt.assert_equals(imc_list[1].size(), 720 * 480)

        dr.image_data = []
        nt.assert_equals(len(dr.image_data), 0)

    @nt.raises(TypeError)
    def test_bad_set_image_data(self):
        dr = DescriptorRequest()
        dr.image_data = "string, not image_data"

    def test_set_and_get_data_location(self):
        dr = DescriptorRequest()

        nt.assert_equals(dr.data_location, "")

        dr.data_location = "first"
        nt.assert_equals(dr.data_location, "first")

        dr.data_location = "second"
        nt.assert_equals(dr.data_location, "second")

        dr.data_location = "42"
        nt.assert_equals(dr.data_location, "42")

        dr.data_location = ""
        nt.assert_equals(dr.data_location, "")

    @nt.raises(TypeError)
    def test_bad_set_data_location(self):
        dr = DescriptorRequest()
        dr.data_location = 5
