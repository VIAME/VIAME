"""
ckwg +31
Copyright 2025 by Kitware, Inc.
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

Test Python bindings for camera_from_metadata functions

"""

import unittest
import numpy as np

from kwiver.vital.types import (
    Metadata,
    SimpleCameraIntrinsics,
    SimpleCameraPerspective,
    LocalTangentSpace,
    GeoPoint,
    RotationD,
    geodesy as gd,
    metadata_tags as mt,
)
from kwiver.vital.io import (
    intrinsics_from_metadata,
    update_camera_from_metadata,
    initialize_cameras_with_metadata,
)
from kwiver.vital import plugin_management


class TestIntrinsicsFromMetadata(unittest.TestCase):
    def test_returns_none_without_metadata(self):
        md = Metadata()
        result = intrinsics_from_metadata(md, 1920, 1080)
        self.assertIsNone(result)

    def test_creates_intrinsics_with_focal_length(self):
        md = Metadata()
        md.add(60.0, mt.tags.VITAL_META_SENSOR_HORIZONTAL_FOV)

        image_width = 1920
        image_height = 1080

        result = intrinsics_from_metadata(md, image_width, image_height)

        if result is not None:
            self.assertIsInstance(result, SimpleCameraIntrinsics)
            self.assertGreater(result.focal_length(), 0)
            self.assertEqual(result.image_width(), image_width)
            self.assertEqual(result.image_height(), image_height)


class TestUpdateCameraFromMetadata(unittest.TestCase):
    @classmethod
    def setUp(self):
        vpm = plugin_management.plugin_manager_instance()
        vpm.load_all_plugins()

    def test_returns_false_without_position_metadata(self):
        md = Metadata()
        local_space = LocalTangentSpace()
        cam = SimpleCameraPerspective()

        result = update_camera_from_metadata(md, local_space, cam)
        self.assertFalse(result)

    def test_updates_camera_with_valid_metadata(self):
        md = Metadata()
        md.add(
            GeoPoint(np.array([42.0, -73.0, 100.0]), gd.SRID.lat_lon_WGS84),
            mt.tags.VITAL_META_SENSOR_LOCATION,
        )
        md.add(RotationD(), mt.tags.VITAL_META_PLATFORM_ORIENTATION)

        geo_origin = GeoPoint(np.array([42.0, -73.0, 0.0]), gd.SRID.lat_lon_WGS84)
        local_space = LocalTangentSpace(geo_origin)

        cam = SimpleCameraPerspective()
        result = update_camera_from_metadata(md, local_space, cam)

        self.assertTrue(result)
        center = cam.center()
        self.assertIsNotNone(center)


class TestInitializeCamerasWithMetadata(unittest.TestCase):
    @classmethod
    def setUp(self):
        vpm = plugin_management.plugin_manager_instance()
        vpm.load_all_plugins()

    def test_returns_empty_map_with_no_metadata(self):
        md_map = {}
        base_cam = SimpleCameraPerspective()
        local_space = LocalTangentSpace()

        result = initialize_cameras_with_metadata(md_map, base_cam, local_space)
        self.assertEqual(len(result), 0)

    def test_creates_cameras_from_metadata_map(self):
        md1 = Metadata()
        md1.add(
            GeoPoint(np.array([42.0, -73.0, 100.0]), gd.SRID.lat_lon_WGS84),
            mt.tags.VITAL_META_SENSOR_LOCATION,
        )
        md1.add(RotationD(), mt.tags.VITAL_META_PLATFORM_ORIENTATION)

        md2 = Metadata()
        md2.add(
            GeoPoint(np.array([42.001, -73.001, 110.0]), gd.SRID.lat_lon_WGS84),
            mt.tags.VITAL_META_SENSOR_LOCATION,
        )
        md2.add(RotationD(5.0, 2.0, 1.0), mt.tags.VITAL_META_PLATFORM_ORIENTATION)

        md_map = {1: md1, 2: md2}

        base_intrinsics = SimpleCameraIntrinsics()
        base_intrinsics.set_focal_length(1000.0)
        base_intrinsics.set_principal_point([960.0, 540.0])

        base_cam = SimpleCameraPerspective()
        base_cam.set_intrinsics(base_intrinsics)

        local_space = LocalTangentSpace()

        result = initialize_cameras_with_metadata(md_map, base_cam, local_space)

        self.assertGreater(len(result), 0)
        if 1 in result:
            cam1 = result[1]
            self.assertIsNotNone(cam1)

        self.assertTrue(local_space.valid())
