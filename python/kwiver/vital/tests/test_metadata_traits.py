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
import nose.tools as nt
import numpy as np

from kwiver.vital.types.metadata_traits import *
from kwiver.vital.types import (
    metadata_tags as mt,
)
from kwiver.vital.tests.cpp_helpers import type_check as tc


class TestVitalMetaTraits(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.tags = [
            mt.tags.VITAL_META_UNKNOWN,
            mt.tags.VITAL_META_METADATA_ORIGIN,
            mt.tags.VITAL_META_UNIX_TIMESTAMP,
            mt.tags.VITAL_META_MISSION_ID,
            mt.tags.VITAL_META_MISSION_NUMBER,
            mt.tags.VITAL_META_PLATFORM_TAIL_NUMBER,
            mt.tags.VITAL_META_PLATFORM_HEADING_ANGLE,
            mt.tags.VITAL_META_PLATFORM_PITCH_ANGLE,
            mt.tags.VITAL_META_PLATFORM_ROLL_ANGLE,
            mt.tags.VITAL_META_PLATFORM_TRUE_AIRSPEED,
            mt.tags.VITAL_META_PLATFORM_INDICATED_AIRSPEED,
            mt.tags.VITAL_META_PLATFORM_DESIGNATION,
            mt.tags.VITAL_META_IMAGE_SOURCE_SENSOR,
            mt.tags.VITAL_META_IMAGE_COORDINATE_SYSTEM,
            mt.tags.VITAL_META_IMAGE_URI,
            mt.tags.VITAL_META_VIDEO_URI,
            mt.tags.VITAL_META_VIDEO_KEY_FRAME,
            mt.tags.VITAL_META_SENSOR_LOCATION,
            mt.tags.VITAL_META_SENSOR_HORIZONTAL_FOV,
            mt.tags.VITAL_META_SENSOR_VERTICAL_FOV,
            mt.tags.VITAL_META_SENSOR_REL_AZ_ANGLE,
            mt.tags.VITAL_META_SENSOR_REL_EL_ANGLE,
            mt.tags.VITAL_META_SENSOR_REL_ROLL_ANGLE,
            mt.tags.VITAL_META_SENSOR_YAW_ANGLE,
            mt.tags.VITAL_META_SENSOR_PITCH_ANGLE,
            mt.tags.VITAL_META_SENSOR_ROLL_ANGLE,
            mt.tags.VITAL_META_SENSOR_TYPE,
            mt.tags.VITAL_META_SLANT_RANGE,
            mt.tags.VITAL_META_TARGET_WIDTH,
            mt.tags.VITAL_META_FRAME_CENTER,
            mt.tags.VITAL_META_CORNER_POINTS,
            mt.tags.VITAL_META_ICING_DETECTED,
            mt.tags.VITAL_META_WIND_DIRECTION,
            mt.tags.VITAL_META_WIND_SPEED,
            mt.tags.VITAL_META_STATIC_PRESSURE,
            mt.tags.VITAL_META_DENSITY_ALTITUDE,
            mt.tags.VITAL_META_OUTSIDE_AIR_TEMPERATURE,
            mt.tags.VITAL_META_TARGET_LOCATION,
            mt.tags.VITAL_META_TARGET_TRK_GATE_WIDTH,
            mt.tags.VITAL_META_TARGET_TRK_GATE_HEIGHT,
            mt.tags.VITAL_META_TARGET_ERROR_EST_CE90,
            mt.tags.VITAL_META_TARGET_ERROR_EST_LE90,
            mt.tags.VITAL_META_DIFFERENTIAL_PRESSURE,
            mt.tags.VITAL_META_PLATFORM_ANG_OF_ATTACK,
            mt.tags.VITAL_META_PLATFORM_VERTICAL_SPEED,
            mt.tags.VITAL_META_PLATFORM_SIDESLIP_ANGLE,
            mt.tags.VITAL_META_AIRFIELD_BAROMET_PRESS,
            mt.tags.VITAL_META_AIRFIELD_ELEVATION,
            mt.tags.VITAL_META_RELATIVE_HUMIDITY,
            mt.tags.VITAL_META_PLATFORM_GROUND_SPEED,
            mt.tags.VITAL_META_GROUND_RANGE,
            mt.tags.VITAL_META_PLATFORM_FUEL_REMAINING,
            mt.tags.VITAL_META_PLATFORM_CALL_SIGN,
            mt.tags.VITAL_META_LASER_PRF_CODE,
            mt.tags.VITAL_META_SENSOR_FOV_NAME,
            mt.tags.VITAL_META_PLATFORM_MAGNET_HEADING,
            mt.tags.VITAL_META_UAS_LDS_VERSION_NUMBER,
            mt.tags.VITAL_META_ANGLE_TO_NORTH,
            mt.tags.VITAL_META_OBLIQUITY_ANGLE,
            mt.tags.VITAL_META_START_DATE_TIME_UTC,
            mt.tags.VITAL_META_EVENT_START_DATE_TIME_UTC,
            mt.tags.VITAL_META_MISSION_START_TIME_UTC,
            mt.tags.VITAL_META_SECURITY_CLASSIFICATION,
            mt.tags.VITAL_META_CLASSIFICATION,
            mt.tags.VITAL_META_SECURITY_LOCAL_MD_SET,
            mt.tags.VITAL_META_WEAPON_LOAD_0601,
            mt.tags.VITAL_META_WEAPON_FIRED_0601,
            mt.tags.VITAL_META_AVERAGE_GSD,
            mt.tags.VITAL_META_GPS_SEC,
            mt.tags.VITAL_META_GPS_WEEK,
            mt.tags.VITAL_META_NORTHING_VEL,
            mt.tags.VITAL_META_EASTING_VEL,
            mt.tags.VITAL_META_UP_VEL,
            mt.tags.VITAL_META_IMU_STATUS,
            mt.tags.VITAL_META_LOCAL_ADJ,
            mt.tags.VITAL_META_DST_FLAGS,
            mt.tags.VITAL_META_RPC_HEIGHT_OFFSET,
            mt.tags.VITAL_META_RPC_HEIGHT_SCALE,
            mt.tags.VITAL_META_RPC_LONG_OFFSET,
            mt.tags.VITAL_META_RPC_LONG_SCALE,
            mt.tags.VITAL_META_RPC_LAT_OFFSET,
            mt.tags.VITAL_META_RPC_LAT_SCALE,
            mt.tags.VITAL_META_RPC_ROW_OFFSET,
            mt.tags.VITAL_META_RPC_ROW_SCALE,
            mt.tags.VITAL_META_RPC_COL_OFFSET,
            mt.tags.VITAL_META_RPC_COL_SCALE,
            mt.tags.VITAL_META_RPC_ROW_NUM_COEFF,
            mt.tags.VITAL_META_RPC_ROW_DEN_COEFF,
            mt.tags.VITAL_META_RPC_COL_NUM_COEFF,
            mt.tags.VITAL_META_RPC_COL_DEN_COEFF,
            mt.tags.VITAL_META_NITF_IDATIM,
            mt.tags.VITAL_META_NITF_BLOCKA_FRFC_LOC_01,
            mt.tags.VITAL_META_NITF_BLOCKA_FRLC_LOC_01,
            mt.tags.VITAL_META_NITF_BLOCKA_LRLC_LOC_01,
            mt.tags.VITAL_META_NITF_BLOCKA_LRFC_LOC_01,
            mt.tags.VITAL_META_NITF_IMAGE_COMMENTS,
            mt.tags.VITAL_META_LAST_TAG,
        ]
        self.name_set = {"Unknown / Undefined entry", "Origin of metadata",
                    "Unix Time Stamp", "Mission ID", "Episode Number",
                    "Platform Tail Number"}

    # One of approx each type is tested
    def test_vital_meta_traits(self):
        self.check_are_valid_traits(VitalMetaTrait_UNKNOWN.name(), "Unknown / Undefined entry",
                                    VitalMetaTrait_UNKNOWN.description(), "Undefined entry",
                                    VitalMetaTrait_UNKNOWN.tag(), self.tags[0],
                                    VitalMetaTrait_UNKNOWN.is_integral(), True,
                                    VitalMetaTrait_UNKNOWN.is_floating_point(), False,
                                    VitalMetaTrait_UNKNOWN.tag_type(), "int")

        self.check_are_valid_traits(VitalMetaTrait_METADATA_ORIGIN.name(), "Origin of metadata",
                                    VitalMetaTrait_METADATA_ORIGIN.description(),
                                    "Name of the metadata standard which was the origin of these metadata vaules if they originated from a video stream. Can be omitted.",
                                    VitalMetaTrait_METADATA_ORIGIN.tag(), self.tags[1],
                                    VitalMetaTrait_METADATA_ORIGIN.is_integral(), False,
                                    VitalMetaTrait_METADATA_ORIGIN.is_floating_point(), False,
                                    VitalMetaTrait_METADATA_ORIGIN.tag_type(), "string")
        type_impl = tc.get_uint64_rep()
        self.check_are_valid_traits(VitalMetaTrait_UNIX_TIMESTAMP.name(), "Unix Time Stamp",
                                    VitalMetaTrait_UNIX_TIMESTAMP.description(), "",
                                    VitalMetaTrait_UNIX_TIMESTAMP.tag(), self.tags[2],
                                    VitalMetaTrait_UNIX_TIMESTAMP.is_integral(), True,
                                    VitalMetaTrait_UNIX_TIMESTAMP.is_floating_point(), False,
                                    VitalMetaTrait_UNIX_TIMESTAMP.tag_type(), type_impl)

        self.check_are_valid_traits(VitalMetaTrait_SLANT_RANGE.name(), "Slant Range (meters)",
                                    VitalMetaTrait_SLANT_RANGE.description(), "Distance to target.",
                                    VitalMetaTrait_SLANT_RANGE.tag(), self.tags[27],
                                    VitalMetaTrait_SLANT_RANGE.is_integral(), False,
                                    VitalMetaTrait_SLANT_RANGE.is_floating_point(), True,
                                    VitalMetaTrait_SLANT_RANGE.tag_type(), "double")

        self.check_are_valid_traits(VitalMetaTrait_VIDEO_KEY_FRAME.name(), "Is frame a key frame",
                                    VitalMetaTrait_VIDEO_KEY_FRAME.description(), "",
                                    VitalMetaTrait_VIDEO_KEY_FRAME.tag(), self.tags[16],
                                    VitalMetaTrait_VIDEO_KEY_FRAME.is_integral(), True,
                                    VitalMetaTrait_VIDEO_KEY_FRAME.is_floating_point(), False,
                                    VitalMetaTrait_VIDEO_KEY_FRAME.tag_type(), "bool")

    def check_are_valid_traits(self, name, exp_name,
                                    desc, exp_desc,
                                    tag, exp_tag,
                                    is_integral, exp_integral,
                                    is_fp, exp_fp,
                                    tag_type, exp_tag_type):
        self.assertEqual(name, exp_name)
        self.assertEqual(desc, exp_desc)
        self.assertEqual(is_integral, exp_integral)
        self.assertEqual(is_fp, exp_fp)
        self.assertEqual(tag_type, exp_tag_type)
        self.assertEqual(int(tag), int(exp_tag))

class TestMetadataTraits(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.tags = [
            mt.tags.VITAL_META_UNKNOWN,
            mt.tags.VITAL_META_METADATA_ORIGIN,
            mt.tags.VITAL_META_UNIX_TIMESTAMP,
            mt.tags.VITAL_META_MISSION_ID,
            mt.tags.VITAL_META_MISSION_NUMBER,
            mt.tags.VITAL_META_PLATFORM_TAIL_NUMBER,
            mt.tags.VITAL_META_PLATFORM_HEADING_ANGLE,
            mt.tags.VITAL_META_PLATFORM_PITCH_ANGLE,
            mt.tags.VITAL_META_PLATFORM_ROLL_ANGLE,
            mt.tags.VITAL_META_PLATFORM_TRUE_AIRSPEED,
            mt.tags.VITAL_META_PLATFORM_INDICATED_AIRSPEED,
            mt.tags.VITAL_META_PLATFORM_DESIGNATION,
            mt.tags.VITAL_META_IMAGE_SOURCE_SENSOR,
            mt.tags.VITAL_META_IMAGE_COORDINATE_SYSTEM,
            mt.tags.VITAL_META_IMAGE_URI,
            mt.tags.VITAL_META_VIDEO_URI,
            mt.tags.VITAL_META_VIDEO_KEY_FRAME,
            mt.tags.VITAL_META_SENSOR_LOCATION,
            mt.tags.VITAL_META_SENSOR_HORIZONTAL_FOV,
            mt.tags.VITAL_META_SENSOR_VERTICAL_FOV,
            mt.tags.VITAL_META_SENSOR_REL_AZ_ANGLE,
            mt.tags.VITAL_META_SENSOR_REL_EL_ANGLE,
            mt.tags.VITAL_META_SENSOR_REL_ROLL_ANGLE,
            mt.tags.VITAL_META_SENSOR_YAW_ANGLE,
            mt.tags.VITAL_META_SENSOR_PITCH_ANGLE,
            mt.tags.VITAL_META_SENSOR_ROLL_ANGLE,
            mt.tags.VITAL_META_SENSOR_TYPE,
            mt.tags.VITAL_META_SLANT_RANGE,
            mt.tags.VITAL_META_TARGET_WIDTH,
            mt.tags.VITAL_META_FRAME_CENTER,
            mt.tags.VITAL_META_CORNER_POINTS,
            mt.tags.VITAL_META_ICING_DETECTED,
            mt.tags.VITAL_META_WIND_DIRECTION,
            mt.tags.VITAL_META_WIND_SPEED,
            mt.tags.VITAL_META_STATIC_PRESSURE,
            mt.tags.VITAL_META_DENSITY_ALTITUDE,
            mt.tags.VITAL_META_OUTSIDE_AIR_TEMPERATURE,
            mt.tags.VITAL_META_TARGET_LOCATION,
            mt.tags.VITAL_META_TARGET_TRK_GATE_WIDTH,
            mt.tags.VITAL_META_TARGET_TRK_GATE_HEIGHT,
            mt.tags.VITAL_META_TARGET_ERROR_EST_CE90,
            mt.tags.VITAL_META_TARGET_ERROR_EST_LE90,
            mt.tags.VITAL_META_DIFFERENTIAL_PRESSURE,
            mt.tags.VITAL_META_PLATFORM_ANG_OF_ATTACK,
            mt.tags.VITAL_META_PLATFORM_VERTICAL_SPEED,
            mt.tags.VITAL_META_PLATFORM_SIDESLIP_ANGLE,
            mt.tags.VITAL_META_AIRFIELD_BAROMET_PRESS,
            mt.tags.VITAL_META_AIRFIELD_ELEVATION,
            mt.tags.VITAL_META_RELATIVE_HUMIDITY,
            mt.tags.VITAL_META_PLATFORM_GROUND_SPEED,
            mt.tags.VITAL_META_GROUND_RANGE,
            mt.tags.VITAL_META_PLATFORM_FUEL_REMAINING,
            mt.tags.VITAL_META_PLATFORM_CALL_SIGN,
            mt.tags.VITAL_META_LASER_PRF_CODE,
            mt.tags.VITAL_META_SENSOR_FOV_NAME,
            mt.tags.VITAL_META_PLATFORM_MAGNET_HEADING,
            mt.tags.VITAL_META_UAS_LDS_VERSION_NUMBER,
            mt.tags.VITAL_META_ANGLE_TO_NORTH,
            mt.tags.VITAL_META_OBLIQUITY_ANGLE,
            mt.tags.VITAL_META_START_DATE_TIME_UTC,
            mt.tags.VITAL_META_EVENT_START_DATE_TIME_UTC,
            mt.tags.VITAL_META_MISSION_START_TIME_UTC,
            mt.tags.VITAL_META_SECURITY_CLASSIFICATION,
            mt.tags.VITAL_META_CLASSIFICATION,
            mt.tags.VITAL_META_SECURITY_LOCAL_MD_SET,
            mt.tags.VITAL_META_AVERAGE_GSD,
            mt.tags.VITAL_META_GPS_SEC,
            mt.tags.VITAL_META_GPS_WEEK,
            mt.tags.VITAL_META_NORTHING_VEL,
            mt.tags.VITAL_META_EASTING_VEL,
            mt.tags.VITAL_META_UP_VEL,
            mt.tags.VITAL_META_IMU_STATUS,
            mt.tags.VITAL_META_LOCAL_ADJ,
            mt.tags.VITAL_META_DST_FLAGS,
            mt.tags.VITAL_META_RPC_HEIGHT_OFFSET,
            mt.tags.VITAL_META_RPC_HEIGHT_SCALE,
            mt.tags.VITAL_META_RPC_LONG_OFFSET,
            mt.tags.VITAL_META_RPC_LONG_SCALE,
            mt.tags.VITAL_META_RPC_LAT_OFFSET,
            mt.tags.VITAL_META_RPC_LAT_SCALE,
            mt.tags.VITAL_META_RPC_ROW_OFFSET,
            mt.tags.VITAL_META_RPC_ROW_SCALE,
            mt.tags.VITAL_META_RPC_COL_OFFSET,
            mt.tags.VITAL_META_RPC_COL_SCALE,
            mt.tags.VITAL_META_RPC_ROW_NUM_COEFF,
            mt.tags.VITAL_META_RPC_ROW_DEN_COEFF,
            mt.tags.VITAL_META_RPC_COL_NUM_COEFF,
            mt.tags.VITAL_META_RPC_COL_DEN_COEFF,
            mt.tags.VITAL_META_NITF_IDATIM,
            mt.tags.VITAL_META_NITF_BLOCKA_FRFC_LOC_01,
            mt.tags.VITAL_META_NITF_BLOCKA_FRLC_LOC_01,
            mt.tags.VITAL_META_NITF_BLOCKA_LRLC_LOC_01,
            mt.tags.VITAL_META_NITF_BLOCKA_LRFC_LOC_01,
            mt.tags.VITAL_META_NITF_IMAGE_COMMENTS,
            mt.tags.VITAL_META_LAST_TAG,
        ]
    def test_constructor(self):
        MetadataTraits()
    def test_find(self):
        s = MetadataTraits()
        meta_trait_found = s.find(self.tags[1])
        self.assertEqual(meta_trait_found.name, "Origin of metadata")
    def test_tag_to_symbol(self):
        s = MetadataTraits()
        self.assertEqual(s.tag_to_symbol(self.tags[1]), "VITAL_META_METADATA_ORIGIN")
    def test_tag_to_name(self):
        s = MetadataTraits()
        self.assertEqual(s.tag_to_name(self.tags[1]), "Origin of metadata")
    def test_tag_to_description(self):
        s = MetadataTraits()
        self.assertEqual(s.tag_to_description(self.tags[0]), "Undefined entry")
