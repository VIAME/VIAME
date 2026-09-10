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

import unittest
import numpy as np

from kwiver.vital.types.metadata_traits import *
from kwiver.vital.types import (
    metadata_tags as mt,
)


class TestVitalMetaTraits(unittest.TestCase):
    # One of approx each type is tested
    def test_vital_meta_traits(self):
        self.check_are_valid_traits(
            tag_traits_by_tag(mt.tags.VITAL_META_UNKNOWN),
            mt.tags.VITAL_META_UNKNOWN,
            "Unknown / Undefined Entry",
            "UNKNOWN",
            "int",
            "Unknown or undefined entry.",
        )

        self.check_are_valid_traits(
            tag_traits_by_tag(mt.tags.VITAL_META_METADATA_ORIGIN),
            mt.tags.VITAL_META_METADATA_ORIGIN,
            "Origin of Metadata",
            "METADATA_ORIGIN",
            "string",
            "Name of the metadata standard used to decode these metadata values from a video stream.",
        )
        self.check_are_valid_traits(
            tag_traits_by_tag(mt.tags.VITAL_META_UNIX_TIMESTAMP),
            mt.tags.VITAL_META_UNIX_TIMESTAMP,
            "Unix Timestamp (microseconds)",
            "UNIX_TIMESTAMP",
            "uint64",
            "Number of microseconds since the Unix epoch, not counting leap seconds.",
        )

        self.check_are_valid_traits(
            tag_traits_by_tag(mt.tags.VITAL_META_SLANT_RANGE),
            mt.tags.VITAL_META_SLANT_RANGE,
            "Slant Range (meters)",
            "SLANT_RANGE",
            "double",
            "Distance to target.",
        )

        self.check_are_valid_traits(
            tag_traits_by_tag(mt.tags.VITAL_META_VIDEO_KEY_FRAME),
            mt.tags.VITAL_META_VIDEO_KEY_FRAME,
            "Is Key Frame",
            "VIDEO_KEY_FRAME",
            "bool",
            "True if the current frame is a key frame.",
        )

        self.check_are_valid_traits(
            tag_traits_by_tag(mt.tags.VITAL_META_SENSOR_ORIENTATION),
            mt.tags.VITAL_META_SENSOR_ORIENTATION,
            "Sensor Orientation",
            "SENSOR_ORIENTATION",
            "rotation_d",
            "Absolute orientation of sensor using North-East-Down (NED) axes.",
        )

    def check_are_valid_traits(
        self, traits, exp_tag, exp_name, exp_enum_name, exp_type, exp_description
    ):
        self.assertEqual(traits.tag(), exp_tag)
        self.assertEqual(traits.name(), exp_name)
        self.assertEqual(traits.enum_name(), exp_enum_name)
        self.assertEqual(traits.type(), exp_type)
        self.assertEqual(traits.description(), exp_description)


class TestMetadataTraits(unittest.TestCase):
    def test_traits_by_tag(self):
        self.assertEqual(
            tag_traits_by_tag(mt.tags.VITAL_META_METADATA_ORIGIN).tag(),
            mt.tags.VITAL_META_METADATA_ORIGIN,
        )

    def test_traits_by_name(self):
        self.assertEqual(
            tag_traits_by_name("Origin of Metadata").tag(),
            mt.tags.VITAL_META_METADATA_ORIGIN,
        )

    def test_traits_by_enum_name(self):
        self.assertEqual(
            tag_traits_by_enum_name("METADATA_ORIGIN").tag(),
            mt.tags.VITAL_META_METADATA_ORIGIN,
        )
