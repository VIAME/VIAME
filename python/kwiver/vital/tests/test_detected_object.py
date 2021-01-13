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

Tests for DetectedObject interface class.

"""
import nose.tools as nt
import numpy as np
import pytest
import unittest

from kwiver.vital.types import (
    BoundingBoxD as BoundingBox,
    descriptor,
    DetectedObject,
    DetectedObjectType,
    Image,
    ImageContainer,
    geodesy,
    GeoPoint,
    Point2d
)

@pytest.mark.skip(reason="See TODO in binding code. Users may experience UB until resolved")
class TestVitalDetectedObject(unittest.TestCase):
    def setUp(self):
        self.loc1 = np.array([-73.759291, 42.849631])
        self.loc2 = np.array([-149.484444, -17.619482])

        self.bbox = BoundingBox(10, 10, 20, 20)
        self.conf = 0.5
        self.dot = DetectedObjectType("example_class", 0.4)
        self.mask = ImageContainer(Image(1080, 720))

        # Values to set outside of constructor
        self.geo_point = GeoPoint(self.loc1, geodesy.SRID.lat_lon_WGS84)
        self.index = 5
        self.detector_name = "example_detector_name"

        self.descriptor = descriptor.new_descriptor(5)
        self.descriptor[:] = 10

        self.note_to_add = "example_note"

        self.keypoint_to_add = Point2d()
        self.keypoint_to_add.value = self.loc2
        self.keypoint_id = "example_keypoint_id"

    def test_init(self):
        DetectedObject(self.bbox)
        DetectedObject(self.bbox, self.conf, self.dot, self.mask)

        # Referencing keyword arguments
        DetectedObject(
            self.bbox, confidence=self.conf, classifications=self.dot, mask=self.mask
        )

    def check_geo_points_equal(self, gp1, gp2):
        if gp1.is_empty() or gp2.is_empty():
            if gp1.is_empty() and gp2.is_empty():
                return True
            return False

        return np.allclose(gp1.location(), gp2.location())

    def check_det_obj_types_equal(self, dot1, dot2):
        if dot1 is None or dot2 is None:
            if dot1 is None and dot2 is None:
                return True
            return False

        dot1_scores = [dot1.score(cname) for cname in dot1.class_names()]
        dot2_scores = [dot2.score(cname) for cname in dot2.class_names()]

        return dot1.class_names() == dot2.class_names() and \
               np.allclose(dot1_scores, dot2_scores)

    def check_img_containers_equal(self, img_c1, img_c2):
        if img_c1 is None or img_c2 is None:
            if img_c1 is None and img_c2 is None:
                return True
            return False

        return np.all(img_c1.asarray() == img_c2.asarray())

    def check_descriptors_equal(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            if desc1 is None and desc2 is None:
                return True
            return False

        return desc1.todoublearray() == desc2.todoublearray()

    def check_keypoints_equal(self, kp1, kp2):
        # Check same keys/number of keys
        keys_equal = set(kp1.keys()) == set(kp2.keys())

        if keys_equal:
            # Check values
            for key in kp1.keys():
                pt1 = kp1[key]
                pt2 = kp2[key]

                values_equal = np.allclose(pt1.value, pt2.value)
                covars_equal = np.allclose(pt1.covariance.matrix(), pt2.covariance.matrix())
                if not (values_equal and covars_equal):
                    return False

            # Iterated through all keys
            return True

        # Keys not equal
        return False

    def check_det_objs_equal(self, do1, do2):
        bboxes_equal = do1.bounding_box == do2.bounding_box
        geo_points_equal = self.check_geo_points_equal(do1.geo_point, do2.geo_point)
        confidences_equal = do1.confidence == do2.confidence
        indices_equal = do1.index == do2.index
        detector_names_equal = do1.detector_name == do2.detector_name
        types_equal = self.check_det_obj_types_equal(do1.type, do2.type)
        masks_equal = self.check_img_containers_equal(do1.mask, do2.mask)
        descriptors_equal = self.check_descriptors_equal(do1.descriptor_copy(), do2.descriptor_copy())
        notes_equal = do1.notes == do2.notes
        keypoints_equal = self.check_keypoints_equal(do1.keypoints, do2.keypoints)

        return bboxes_equal and geo_points_equal and confidences_equal and \
               indices_equal and detector_names_equal and types_equal and \
               masks_equal and descriptors_equal and notes_equal and keypoints_equal

    def test_nice_format(self):
        # Test default
        do = DetectedObject(self.bbox)
        nt.assert_equal(do.__nice__(), "conf=1.0")

        do.confidence = -0.5
        nt.assert_equal(do.__nice__(), "conf=-0.5")

    def test_repr_format(self):
        # Test default
        do = DetectedObject(self.bbox)
        nt.assert_equal(do.__repr__(), "<DetectedObject(conf=1.0) at {}>".format(hex(id(do))))

        do = DetectedObject(self.bbox, confidence=-0.5)
        nt.assert_equal(do.__repr__(), "<DetectedObject(conf=-0.5) at {}>".format(hex(id(do))))

    def test_str_format(self):
        # Test default
        do = DetectedObject(self.bbox)
        nt.assert_equal(do.__str__(), "<DetectedObject(conf=1.0)>")

        do = DetectedObject(self.bbox, confidence=-0.5)
        nt.assert_equal(do.__str__(), "<DetectedObject(conf=-0.5)>")

    def test_clone(self):
        do = DetectedObject(self.bbox)
        do_clone = do.clone()
        nt.ok_(self.check_det_objs_equal(do, do_clone))

        do = DetectedObject(self.bbox, self.conf, self.dot, self.mask)
        do_clone = do.clone()
        nt.ok_(self.check_det_objs_equal(do, do_clone))

        # Try setting some values
        do.geo_point = self.geo_point
        do.index = self.index
        do.detector_name = self.detector_name
        do.set_descriptor(self.descriptor)
        do.add_note(self.note_to_add)

        do.add_keypoint(self.keypoint_id, self.keypoint_to_add)
        # First show that its a deep copy. Should no longer be equal
        nt.assert_false(self.check_det_objs_equal(do, do_clone))

        # Now clone
        do_clone = do.clone()
        nt.ok_(self.check_det_objs_equal(do, do_clone))

    def test_get_set_bbox(self):
        do = DetectedObject(self.bbox)

        # Check default
        nt.ok_(do.bounding_box, self.bbox)

        # Setting to different value
        new_bbox = BoundingBox(20, 20, 40, 40)
        do.bounding_box = new_bbox
        nt.ok_(do.bounding_box == new_bbox)


    def test_get_set_geo_point(self):
        do = DetectedObject(self.bbox)

        # Check default
        nt.ok_(self.check_geo_points_equal(do.geo_point, GeoPoint()))

        # Setting to different value
        do.geo_point = self.geo_point
        nt.ok_(self.check_geo_points_equal(do.geo_point, self.geo_point))

    def test_get_set_confidence(self):
        # Check default
        do = DetectedObject(self.bbox)
        np.testing.assert_almost_equal(do.confidence, 1.0)

        # Check setting through constructor
        do = DetectedObject(self.bbox, confidence=-1.5)
        np.testing.assert_almost_equal(do.confidence, -1.5)

        # Check setting through setter
        do.confidence = 2.5
        np.testing.assert_almost_equal(do.confidence, 2.5)

    def test_get_set_index(self):
        # Check default
        do = DetectedObject(self.bbox)
        nt.assert_equal(do.index, 0)

        do.index = 5
        nt.assert_equal(do.index, 5)

    def test_get_set_detector_name(self):
        # Check default
        do = DetectedObject(self.bbox)
        nt.assert_equal(do.detector_name, "")

        do.detector_name = self.detector_name
        nt.assert_equal(do.detector_name, self.detector_name)

    def test_get_set_type(self):
        # Check default
        do = DetectedObject(self.bbox)
        nt.ok_(self.check_det_obj_types_equal(do.type, None))

        # Check setting through setter
        do.type = self.dot
        nt.ok_(self.check_det_obj_types_equal(do.type, self.dot))

        # Check setting through constructor
        new_dot = DetectedObjectType("other_example_class", -3.14)
        do = DetectedObject(self.bbox, classifications=new_dot)
        nt.ok_(self.check_det_obj_types_equal(do.type, new_dot))

    # See TODO in binding code for an explanation of why these are
    # commented out
    def test_get_set_mask(self):
        # Check default
        do = DetectedObject(self.bbox)
        nt.ok_(self.check_img_containers_equal(do.mask, None))

        # Check setting through setter
        do.mask = self.mask
        nt.ok_(self.check_img_containers_equal(do.mask, self.mask))

        # Check setting through constructor
        new_mask = ImageContainer(Image(2048, 1080))
        do = DetectedObject(self.bbox, mask=new_mask)
        nt.ok_(self.check_img_containers_equal(do.mask, new_mask))

    def test_descriptor(self):
        # Check default
        do = DetectedObject(self.bbox)
        nt.ok_(self.check_descriptors_equal(do.descriptor_copy(), None))

        do.set_descriptor(self.descriptor)
        nt.ok_(self.check_descriptors_equal(do.descriptor_copy(), self.descriptor))

    # In the C++ class, a pointer to a const descriptor is stored/exposed
    # Pybind casts away all const-ness, which would result in undefined behavior if the
    # member wasn't copied in the binding code.
    def test_descriptor_modify(self):
        do = DetectedObject(self.bbox)
        do.set_descriptor(self.descriptor)

        # Attempts to modify the descriptor don't work
        do.descriptor_copy()[0] += 1
        # print(do.descriptor_copy().todoublearray(), self.descriptor.todoublearray())
        nt.ok_(self.check_descriptors_equal(do.descriptor_copy(), self.descriptor))

        # Modify the object copied from. Changes should not be reflected in
        # the detected_objects reference
        self.descriptor[0] += 1
        # print(do.descriptor_copy().todoublearray(), self.descriptor.todoublearray())
        nt.assert_false(self.check_descriptors_equal(do.descriptor_copy(), self.descriptor))

        # Storing the copy in a new variable, obviously, allows for modification
        desc = do.descriptor_copy()
        desc[0] += 1
        nt.assert_false(self.check_descriptors_equal(do.descriptor_copy(), desc))

    def test_notes(self):
        # Check default
        do = DetectedObject(self.bbox)
        nt.assert_equal(do.notes, [])

        # Clearing empty is OK
        do.clear_notes()
        nt.assert_equal(do.notes, [])

        # Add a few values
        do.add_note(self.note_to_add)
        exp_notes = [self.note_to_add]
        nt.assert_equal(do.notes, exp_notes)

        new_note = "other_example_note"
        do.add_note(new_note)
        exp_notes.append(new_note)
        nt.assert_equal(do.notes, exp_notes)

        # Clearing works as expected
        do.clear_notes()
        nt.assert_equal(do.notes, [])

    def test_keypoints(self):
        # Check default
        do = DetectedObject(self.bbox)
        self.check_keypoints_equal(do.keypoints, dict())

        # Clearing empty is OK
        do.clear_keypoints()
        self.check_keypoints_equal(do.keypoints, dict())

        # Add a few values
        do.add_keypoint(self.keypoint_id, self.keypoint_to_add)
        exp_keypoints = {self.keypoint_id: self.keypoint_to_add}
        self.check_keypoints_equal(do.keypoints, exp_keypoints)

        new_keypoint_id = "other_example_keypoint_id"
        new_keypoint = Point2d()
        new_keypoint.value = self.loc1
        do.add_keypoint(new_keypoint_id, new_keypoint)
        exp_keypoints[new_keypoint_id] = new_keypoint
        self.check_keypoints_equal(do.keypoints, exp_keypoints)

        # Clearing works as expected
        do.clear_keypoints()
        self.check_keypoints_equal(do.keypoints, dict())
