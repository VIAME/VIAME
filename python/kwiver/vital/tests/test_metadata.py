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

Tests for the vital classes metadata_item, typed_metadata (templated),
unknown_metadata_item, and metadata

"""

import numpy as np
import sys
import unittest
import platform

from kwiver.vital.types.metadata_traits import *
from kwiver.vital.types.metadata import *
from kwiver.vital.types import (
    geodesy as gd,
    GeoPoint,
    GeoPolygon,
    metadata_tags as mt,
    Polygon,
    Timestamp,
)
from kwiver.vital.tests.cpp_helpers import type_check as tc


# Helper class that stores a typed_metadata object's
# name, tag, and data properties
class PropInfo(object):
    def __init__(self, name, tag, data):
        self.name = name
        self.tag = tag
        self.data = data


# Helper class that stores which types a typed_metadata
# can be converted to
class TypeInfo(object):
    def __init__(
        self,
        typename,
        has_double=False,
        has_uint64=False,
        has_string=False,
        as_double=None,
        as_uint64=None,
        as_string=None,
    ):
        self.typename = typename
        self.has_double = has_double
        self.has_uint64 = has_uint64
        self.has_string = has_string

        self.as_double = as_double
        self.as_uint64 = as_uint64
        self.as_string = as_string


class TestVitalMetadataItemSubclasses(unittest.TestCase):
    # Creates a geo_point and geo_poly
    def setUp(self):
        self.g_point = GeoPoint(np.array([1, 2, 3]), gd.SRID.lat_lon_WGS84)

        poly = Polygon(
            [
                np.array([10, 10]),
                np.array([10, 50]),
                np.array([50, 50]),
                np.array([30, 30]),
            ]
        )

        self.g_poly = GeoPolygon(poly, gd.SRID.lat_lon_WGS84)

    # We're going to test 1 TypedMetadata of each type, plus
    # TypedMetadata_UNKNOWN
    def test_driver(self):
        # GPS_WEEK - int
        tag = mt.tags.VITAL_META_GPS_WEEK
        inst1 = MetadataItem(tag, 3)
        inst2 = MetadataItem(tag, -3)
        type_info1 = TypeInfo("int", as_string="3")
        type_info2 = TypeInfo("int", as_string="-3")
        self.check_instance(inst1, PropInfo("GPS Week", tag, 3), type_info1)
        self.check_instance(inst2, PropInfo("GPS Week", tag, -3), type_info2)

        # UNIX_TIMESTAMP - uint64
        tag = mt.tags.VITAL_META_UNIX_TIMESTAMP
        inst1 = MetadataItem(tag, 5)
        inst2 = MetadataItem(tag, 0)
        prop_info1 = PropInfo("Unix Timestamp (microseconds)", tag, 5)
        prop_info2 = PropInfo("Unix Timestamp (microseconds)", tag, 0)
        type_impl = tc.get_uint64_rep()
        type_info1 = TypeInfo(type_impl, has_uint64=True, as_uint64=5, as_string="5")
        type_info2 = TypeInfo(type_impl, has_uint64=True, as_uint64=0, as_string="0")
        self.check_instance(inst1, prop_info1, type_info1)
        self.check_instance(inst2, prop_info2, type_info2)

        # SLANT_RANGE - double
        tag = mt.tags.VITAL_META_SLANT_RANGE
        inst1 = MetadataItem(tag, 3.14)
        inst2 = MetadataItem(tag, -3.14)
        prop_info1 = PropInfo("Slant Range (meters)", tag, 3.14)
        prop_info2 = PropInfo("Slant Range (meters)", tag, -3.14)
        type_info1 = TypeInfo(
            "double", has_double=True, as_double=3.14, as_string="3.14"
        )
        type_info2 = TypeInfo(
            "double", has_double=True, as_double=-3.14, as_string="-3.14"
        )
        self.check_instance(inst1, prop_info1, type_info1)
        self.check_instance(inst2, prop_info2, type_info2)

        # MISSION_ID - string
        tag = mt.tags.VITAL_META_MISSION_ID
        inst1 = MetadataItem(tag, "")
        inst2 = MetadataItem(tag, "data123")
        prop_info1 = PropInfo("Mission ID", tag, "")
        prop_info2 = PropInfo("Mission ID", tag, "data123")
        type_info1 = TypeInfo("string", has_string=True, as_string="")
        type_info2 = TypeInfo("string", has_string=True, as_string="data123")
        self.check_instance(inst1, prop_info1, type_info1)
        self.check_instance(inst2, prop_info2, type_info2)

        # VIDEO_KEY_FRAME - bool
        tag = mt.tags.VITAL_META_VIDEO_KEY_FRAME
        inst1 = MetadataItem(tag, True)
        inst2 = MetadataItem(tag, False)
        type_info1 = TypeInfo("bool", as_string="True")
        type_info2 = TypeInfo("bool", as_string="False")
        self.check_instance(inst1, PropInfo("Is Key Frame", tag, True), type_info1)
        self.check_instance(inst2, PropInfo("Is Key Frame", tag, False), type_info2)

        # FRAME_CENTER - kwiver::vital::geo_point
        tag = mt.tags.VITAL_META_FRAME_CENTER
        inst1 = MetadataItem(tag, GeoPoint())
        inst2 = MetadataItem(tag, self.g_point)
        if platform.system() == "Windows":
            # std::type_info::name (used to derive typeinfo internally) result
            # is different on windows versus the result on linux after
            # demangling. This is expected.
            type_info1 = TypeInfo(
                "class kwiver::vital::geo_point", as_string=str(GeoPoint())
            )
            type_info2 = TypeInfo(
                "class kwiver::vital::geo_point", as_string=str(self.g_point)
            )
        else:
            type_info1 = TypeInfo("kwiver::vital::geo_point", as_string=str(GeoPoint()))
            type_info2 = TypeInfo(
                "kwiver::vital::geo_point", as_string=str(self.g_point)
            )
        self.check_instance(
            inst1,
            PropInfo("Geodetic Frame Center (lon/lat/alt)", tag, GeoPoint()),
            type_info1,
        )
        self.check_instance(
            inst2,
            PropInfo("Geodetic Frame Center (lon/lat/alt)", tag, self.g_point),
            type_info2,
        )

        # CORNER_POINTS - kwiver::vital::geo_polygon
        tag = mt.tags.VITAL_META_CORNER_POINTS
        inst1 = MetadataItem(tag, GeoPolygon())
        inst2 = MetadataItem(tag, self.g_poly)
        if platform.system() == "Windows":
            # std::type_info::name (used to derive typeinfo internally) result
            # is different on windows versus the result on linux after
            # demangling. This is expected.
            type_info1 = TypeInfo(
                "class kwiver::vital::geo_polygon", as_string=str(GeoPolygon())
            )
            type_info2 = TypeInfo(
                "class kwiver::vital::geo_polygon", as_string=str(self.g_poly)
            )
        else:
            type_info1 = TypeInfo(
                "kwiver::vital::geo_polygon", as_string=str(GeoPolygon())
            )
            type_info2 = TypeInfo(
                "kwiver::vital::geo_polygon", as_string=str(self.g_poly)
            )
        self.check_instance(
            inst1, PropInfo("Corner Points (lon/lat)", tag, GeoPolygon()), type_info1
        )
        self.check_instance(
            inst2, PropInfo("Corner Points (lon/lat)", tag, self.g_poly), type_info2
        )

        # UNKNOWN - int
        tag = mt.tags.VITAL_META_UNKNOWN
        inst1 = MetadataItem(tag, 2)
        inst2 = MetadataItem(tag, -2)
        type_info1 = TypeInfo("int", as_string="2")
        type_info2 = TypeInfo("int", as_string="-2")
        self.check_instance(
            inst1, PropInfo("Unknown / Undefined Entry", tag, 2), type_info1, False
        )
        self.check_instance(
            inst2, PropInfo("Unknown / Undefined Entry", tag, -2), type_info2, False
        )

    def check_instance(self, inst, prop_info, type_info, is_valid=True):
        self.check_initial_properties(inst, prop_info)
        self.check_as_double(inst, type_info.has_double, type_info.as_double)
        self.check_as_uint64(inst, type_info.has_uint64, type_info.as_uint64)
        self.check_as_string(inst, type_info.has_string, type_info.as_string)
        self.check_is_valid(inst, is_valid)
        self.check_typename(inst, type_info.typename)

    def check_initial_properties(self, inst, prop_info):
        self.assertEqual(inst.name, prop_info.name)
        self.assertEqual(inst.tag, prop_info.tag)

        # A few tests on inst.data
        self.assertTrue(isinstance(inst.data, type(prop_info.data)))
        if isinstance(prop_info.data, GeoPoint):
            self.assertEqual(inst.data.crs(), prop_info.data.crs())
            if prop_info.data.is_empty():
                self.assertTrue(inst.data.is_empty())
            else:
                inst_loc = inst.data.location()
                exp_loc = prop_info.data.location()
                np.testing.assert_array_almost_equal(inst_loc, exp_loc)

        elif isinstance(prop_info.data, GeoPolygon):
            self.assertEqual(inst.data.crs(), prop_info.data.crs())
            if prop_info.data.is_empty():
                self.assertTrue(inst.data.is_empty())
            else:
                inst_verts = inst.data.polygon().get_vertices()
                exp_verts = prop_info.data.polygon().get_vertices()
                np.testing.assert_array_almost_equal(inst_verts, exp_verts)

        elif prop_info.data is None:
            self.assertTrue(inst.data is None)

        else:
            self.assertEqual(inst.data, prop_info.data)

    def check_is_valid(self, inst, is_valid):
        self.assertEqual(bool(inst), is_valid)
        self.assertEqual(inst.is_valid(), is_valid)
        self.assertEqual(inst.__nonzero__(), is_valid)
        self.assertEqual(inst.__bool__(), is_valid)

    # Following 3 functions test whether the conversion functions work
    # as expected. If the conversion is possible, test that the data matches
    # Otherwise, check that the proper exception is thrown.
    def check_as_double(self, inst, exp_has_double, exp_val):
        self.assertEqual(inst.has_double(), exp_has_double)
        if inst.has_double() or exp_val is not None:
            # Make sure that the value we're about to compare is correct type
            np.testing.assert_almost_equal(inst.as_double(), exp_val)
        else:
            with self.assertRaises(RuntimeError):
                inst.as_double()

    def check_as_uint64(self, inst, exp_has_uint64, exp_val):
        self.assertEqual(inst.has_uint64(), exp_has_uint64)
        if inst.has_uint64() or exp_val is not None:
            # Make sure that the value we're about to compare is correct type
            self.assertEqual(inst.as_uint64(), exp_val)
        else:
            with self.assertRaises(RuntimeError):
                inst.as_uint64()

    # as_string is defined for every type
    def check_as_string(self, inst, exp_has_string, exp_val):
        self.assertEqual(inst.has_string(), exp_has_string)
        # Always able to convert
        self.assertEqual(inst.as_string(), exp_val)

    def check_typename(self, inst, typename):
        self.assertEqual(inst.type, typename)


# And finally, the actual metadata class
class TestVitalMetadata(unittest.TestCase):
    def setUp(self):
        self.tags = [
            mt.tags.VITAL_META_GPS_WEEK,
            mt.tags.VITAL_META_UNIX_TIMESTAMP,
            mt.tags.VITAL_META_SLANT_RANGE,
            mt.tags.VITAL_META_MISSION_ID,
            mt.tags.VITAL_META_VIDEO_KEY_FRAME,
            mt.tags.VITAL_META_FRAME_CENTER,
            mt.tags.VITAL_META_CORNER_POINTS,
            mt.tags.VITAL_META_UNKNOWN,
        ]
        self.small_tag = [
            mt.tags.VITAL_META_UNKNOWN,
            mt.tags.VITAL_META_UNIX_TIMESTAMP,
            mt.tags.VITAL_META_SLANT_RANGE,
            mt.tags.VITAL_META_MISSION_ID,
            mt.tags.VITAL_META_VIDEO_KEY_FRAME,
        ]

    def populate_metadata(self, m):
        m.add(100, self.tags[-1])
        m.add("hello", self.tags[3])
        m.add(2, self.tags[1])
        m.add(True, self.tags[4])
        m.add(1.1, self.tags[2])

    def test_init(self):
        Metadata()

    def test_add(self):
        m = Metadata()
        self.assertEqual(m.size(), 0)
        self.assertTrue(m.empty())
        m.add(0, self.tags[-1])
        m.add("hello", self.tags[3])
        m.add(2, self.tags[1])
        m.add(True, self.tags[4])
        self.assertEqual(m.size(), 4)

    def test_erase(self):
        m = Metadata()
        for tag in self.tags:
            self.assertFalse(m.erase(tag))
        self.populate_metadata(m)
        for tag in self.small_tag:
            self.assertTrue(m.erase(tag))

    # Tests for has and find
    def test_retrieve(self):
        m = Metadata()
        # Make sure there are no initial elements
        for tag in self.tags:
            self.assertFalse(m.has(tag))
            self.assertFalse(m.find(tag).is_valid())
        self.populate_metadata(m)
        uint64_type_impl = tc.get_uint64_rep()
        possible_types = {"int", "bool", uint64_type_impl, "string", "double"}
        for tag in self.small_tag:
            self.assertTrue(m.has(tag))
            self.assertTrue(isinstance(m.find(tag), MetadataItem))
            found = m.find(tag)
            self.assertIn(found.type, possible_types)

    def test_timestamp(self):
        m = Metadata()
        self.populate_metadata(m)
        metadatas = [Metadata(), m]
        for m in metadatas:
            self.assertFalse(m.timestamp.is_valid())

            t = Timestamp()

            t.set_time_seconds(1234)
            m.timestamp = t
            self.assertEqual(m.timestamp.get_time_seconds(), 1234)
            self.assertFalse(m.timestamp.has_valid_frame())

            t.set_frame(1)
            m.timestamp = t
            self.assertTrue(m.timestamp == t)
