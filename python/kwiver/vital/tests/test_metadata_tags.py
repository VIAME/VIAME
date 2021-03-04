"""
ckwg +31
Copyright 2020 by Kitware Inc.
All rights reserved.

Redistribution and use in source and binary forms with or without
modification are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES INCLUDING BUT NOT LIMITED TO THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT INDIRECT INCIDENTAL SPECIAL EXEMPLARY OR CONSEQUENTIAL
DAMAGES (INCLUDING BUT NOT LIMITED TO PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE DATA OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY WHETHER IN CONTRACT STRICT LIABILITY
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

Tests for Python interface to vital::metadata_tags

"""

from kwiver.vital.types import metadata_tags as mt

import nose.tools as nt


class TestVitalMetadataTags(object):
    def setUp(self):
        # This gets us a list of every tag enum value,
        # including the VITAL_META_LAST_TAG
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

    def test_generated_enums(self):
        expected_val = 0
        for t in self.tags:
            nt.assert_equals(
                int(t),
                expected_val,
                "Enum mismatch for {}. Expected {}, got {}".format(
                    str(t), expected_val, int(t)
                ),
            )

            # Print out every 20th element, as well as the last element
            if expected_val % 20 == 0 or expected_val == len(self.tags)-1:
                print(str(t), "has value", int(t), "which matches expected value of", expected_val)

            expected_val += 1
