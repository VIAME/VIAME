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

Tests for Python interface to vital::feature_track_set/state

"""

import unittest
import nose.tools as nt
import numpy as np
from kwiver.vital.tests.py_helpers import no_call_pure_virtual_method
from kwiver.vital.types import (
    FeatureF,
    Descriptor,
    FeatureTrackState as ftstate,
    FeatureTrackSet as ftset,
    TrackSet,
    Track,
    TrackDescriptor,
    DescriptorSet,
    new_descriptor,
    TrackState,
    SimpleFeatureSet as sfs,
)

"""
Test Feature Track State
"""
class TestFeatureTrackState(unittest.TestCase):
    @classmethod
    def setUp(self):
      self.f1 = FeatureF([1, 1], 1, 2, 1)
      self.desc = new_descriptor(33, 'd')
      self.ts = TrackState(0)
    def test_constructors(self):
      ftstate(13, self.f1, self.desc)
      ftstate(ftstate(13, self.f1, self.desc))
    def test_members(self):
      test_ft = ftstate(13, self.f1, self.desc)
      test_ft.inlier = True
      self.assertTrue(test_ft.inlier)
      self.assertEqual(test_ft.frame_id, 13)
      nt.assert_equal(test_ft.feature, self.f1)
      nt.assert_equal(test_ft.descriptor, self.desc)
    def test_methods(self):
      test_ft = ftstate(13, self.f1, self.desc)
      test_ft_clone = test_ft.clone()
      nt.ok_(isinstance(test_ft_clone, ftstate))
      nt.assert_equal(test_ft_clone.frame_id, test_ft.frame_id)
      test_ft_downcast = test_ft.downcast()
      nt.ok_(isinstance(test_ft_downcast, ftstate))
      nt.assert_equal(test_ft_downcast.frame_id, test_ft.frame_id)

"""
Test Feature Track Set
"""
class TestFeatureTrackSet(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.track_state = TrackState(15)
        self.track = Track(23)
        self.track.append(TrackState(1))
        self.track.append(self.track_state)
        self._track_arr = [Track(15), Track(1),
                                    Track(150), Track(9),
                                    self.track]
    def test_construct(self):
      ftset()
      ftset(self._track_arr)
    def test_methods(self):
      test_feat_set = ftset(self._track_arr)
      self.assertEqual(test_feat_set.all_frame_ids(), {1, 15})
      nt.assert_equal(test_feat_set.get_track(1), self._track_arr[1] )
      self.assertEqual(0, test_feat_set.first_frame())
      self.assertEqual(15, test_feat_set.last_frame())
      self.assertEqual(5, len(test_feat_set.tracks()))
      self.assertEqual(len(test_feat_set), 5)
      cloned_test_feat_set = test_feat_set.clone()
      self.assertIsInstance(cloned_test_feat_set, ftset)
      self.assertEqual(cloned_test_feat_set.all_frame_ids(), test_feat_set.all_frame_ids())
      self.assertIsInstance(test_feat_set.last_frame_features(), sfs)
      self.assertIsInstance(test_feat_set.last_frame_descriptors(), DescriptorSet)
      self.assertIsInstance(test_feat_set.frame_features(), sfs)
      self.assertIsInstance(test_feat_set.frame_descriptors(), DescriptorSet)
      track_state_list = test_feat_set.frame_feature_track_states(15)
      self.assertListEqual(track_state_list, [])
      self.assertEqual(test_feat_set.keyframes(), set())
"""
Implement and Test Class Inheriting from Feature Track Set
"""
class SubFeatureTrackSet(ftset):
  def __init__(self):
    ftset.__init__(self)
    self.tracks = [15, 1, 18, 9]
  def last_frame_features(self):
    return sfs()
  def last_frame_descriptors(self):
    return DescriptorSet()
  def frame_features(self, offset):
    return sfs()
  def frame_descriptors(self, offset):
    return DescriptorSet()
  def frame_feature_track_states(self, offset):
    return []
  def keyframes(self):
    return set()
  def clone(self):
    return SubFeatureTrackSet()
  def size(self):
    return len(self.tracks)

class TestSubFeatureTrackSet(unittest.TestCase):
  def test_constructors(self):
    SubFeatureTrackSet()
  def test_overridden_methods(self):
    tst = SubFeatureTrackSet()
    self.assertIsInstance(tst.last_frame_features(), sfs)
    self.assertIsInstance(tst.last_frame_descriptors(), DescriptorSet)
    self.assertListEqual([], tst.frame_feature_track_states(15))
    self.assertEqual(set(), tst.keyframes())
    self.assertIsInstance(tst.frame_features(0), sfs)
    self.assertIsInstance(tst.frame_descriptors(0), DescriptorSet)
  def test_inherited_methods(self):
    tst = SubFeatureTrackSet()
    cloned = tst.clone()
    self.assertIsInstance(cloned, ftset)
    self.assertListEqual(cloned.tracks, tst.tracks)
    self.assertEqual(0, len(tst))
    self.assertEqual(4, tst.size())
