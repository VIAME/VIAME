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

Tests for Python interface to vital::sfm_constraints

"""

import nose.tools as nt
import unittest
import numpy as np
from kwiver.vital.modules import modules
from kwiver.vital.types.metadata import *
from kwiver.vital.types.metadata_traits import *
from kwiver.vital.types import (
    Metadata,
    LocalGeoCS,
    rotation,
    RotationD,
    RotationF,
    SFMConstraints,
    geodesy,
    GeoPoint,
    metadata_tags as mt,
    SimpleMetadataMap,
)

modules.load_known_modules()
class TestSFMConstraints(unittest.TestCase):
    @classmethod
    def setUp(self):
      self.meta_ = SimpleMetadataMap()
      self.geo_ = LocalGeoCS()
      self.small_tag = [
            mt.tags.VITAL_META_UNKNOWN,
            mt.tags.VITAL_META_UNIX_TIMESTAMP,
            mt.tags.VITAL_META_SLANT_RANGE,
            mt.tags.VITAL_META_MISSION_ID,
            mt.tags.VITAL_META_VIDEO_KEY_FRAME,
        ]
      self.loc1 = np.array([-73.759291, 42.849631])
      self.crs_ll = geodesy.SRID.lat_lon_WGS84
      self.geo_pt1_ = GeoPoint(self.loc1, self.crs_ll)
      self.geo_.geo_origin = self.geo_pt1_
    def test_init(self):
      s = SFMConstraints()
      SFMConstraints(s)
      SFMConstraints(self.meta_, self.geo_)
    def test_properties(self):
      # modules.load_known_modules()
      # metadata property
      s = SFMConstraints(self.meta_, self.geo_)
      get_meta = s.metadata
      nt.assert_equal(get_meta.size(), 0)
      m = SimpleMetadataMap()
      s.metadata = m
      nt.assert_equal(s.metadata.size(), 0)

      # local_geo_property
      ret_geo = s.local_geo_cs
      np.testing.assert_array_almost_equal(ret_geo.geo_origin.location(self.crs_ll),
                                                                        self.geo_pt1_.location())
      s = SFMConstraints()
      s.local_geo_cs = self.geo_
      ret_geo = s.local_geo_cs
      np.testing.assert_array_almost_equal(ret_geo.geo_origin.location(self.crs_ll),
                                                                        self.geo_pt1_.location())

    def test_get_camera_position_prior_local(self):
      s = SFMConstraints(self.meta_, self.geo_)
      nt.assert_false(s.get_camera_position_prior_local(0, np.array([0, 1, 3])))
      nt.assert_false(s.get_camera_position_prior_local(0, RotationD([1, 2, 3, 4])))
    def test_camera_position_priors(self):
      s = SFMConstraints(self.meta_, self.geo_)
      nt.assert_dict_equal(s.get_camera_position_priors(), {})
    def test_image_properties(self):
      s = SFMConstraints(self.meta_, self.geo_)
      s.store_image_size(0, 1080, 720)
      a,b = 0,0
      founda, foundb = False, False
      founda, a = s.get_image_width(0, a)
      foundb, b = s.get_image_height(0, b)
      nt.ok_(founda)
      nt.ok_(foundb)
      nt.assert_equal(a, 1080)
      nt.assert_equal(b, 720)
      found_focal = True
      focal_len = 0.1
      found_focal, focal_len = s.get_focal_length_prior(0, focal_len)
      nt.assert_false(found_focal)
      nt.assert_almost_equal(focal_len, 0.1)
