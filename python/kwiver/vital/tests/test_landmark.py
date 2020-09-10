"""
ckwg +31
Copyright 2016-2020 by Kitware, Inc.
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

Tests for Landmark interface

"""
from __future__ import print_function
import unittest

import nose.tools
import numpy
import numpy.testing as npt
from kwiver.vital.tests.py_helpers import no_call_pure_virtual_method
from kwiver.vital.tests.cpp_helpers import landmark_helpers as lh
from kwiver.vital.types import Landmark, LandmarkF, LandmarkD, Covar3f, Covar3d, RGBColor

class TestBaseLandmark(unittest.TestCase):
    def test_init(self):
        Landmark()
    def test_methods(self):
        no_call_pure_virtual_method(Landmark().data_type)
        no_call_pure_virtual_method(Landmark().loc)
        no_call_pure_virtual_method(Landmark().scale)
        no_call_pure_virtual_method(Landmark().normal)
        no_call_pure_virtual_method(Landmark().covar)
        no_call_pure_virtual_method(Landmark().color)
        no_call_pure_virtual_method(Landmark().observations)
        no_call_pure_virtual_method(Landmark().cos_obs_angle)

class SubLandmark(Landmark):
    def __init__(self):
        Landmark.__init__(self)
    def clone(self):
        return SubLandmark()
    def data_type(self):
        return "sub"
    def loc(self):
        return [2, 5, 7]
    def scale(self):
        return 0.2
    def normal(self):
        return [9, 21, 13]
    def covar(self):
        return Covar3d(7)
    def color(self):
        return RGBColor(12, 240, 120)
    def observations(self):
        return 3
    def cos_obs_angle(self):
        return 0.02

class TestSubLandmark(unittest.TestCase):
    def test_inheritance(self):
        nose.tools.ok_(issubclass(SubLandmark, Landmark))

    def test_overrides(self):
        sl = SubLandmark()
        sl_clone = lh.call_clone(sl)
        nose.tools.ok_(isinstance(sl_clone, SubLandmark))
        nose.tools.assert_equal(sl_clone.data_type(), sl.data_type())
        nose.tools.assert_equal(lh.call_scale(sl), 0.2)
        nose.tools.assert_equal(lh.call_cos_obs_angle(sl), 0.02)
        npt.assert_equal(lh.call_loc(sl), [2, 5, 7])
        npt.assert_equal(lh.call_normal(sl), [9, 21, 13])
        npt.assert_equal(lh.call_covar(sl).matrix(), Covar3d(7).matrix())


class TestLandmarks (unittest.TestCase):

    precs = [15, 6]

    def test_new(self):
        LandmarkF()
        LandmarkD()
        LandmarkF([1, 1, 1])
        LandmarkD([1, 1, 1])
        LandmarkF([1, 1, 1], scale=10)
        LandmarkD([1, 1, 1], scale=10)

    def test_type_name(self):
        # Double default
        nose.tools.assert_equal(LandmarkD().data_type, 'double')
        nose.tools.assert_equal(LandmarkF().data_type, 'float')

        nose.tools.assert_equal(LandmarkD([1, 2, 2]).data_type, 'double')
        nose.tools.assert_equal(LandmarkF([1, 2, 2]).data_type, 'float')

    def test_get_loc(self):
        l = LandmarkF()
        numpy.testing.assert_equal(l.loc, [0,0,0])

        l = LandmarkF([1, 2, 3])
        numpy.testing.assert_equal(l.loc, [1,2,3])

        l = LandmarkD()
        numpy.testing.assert_equal(l.loc, [0,0,0])

        l = LandmarkD([1, 2, 3])
        numpy.testing.assert_equal(l.loc, [1,2,3])

    def test_set_loc(self):
        l = LandmarkF()
        l.loc = [1,1,1]
        numpy.testing.assert_equal(l.loc, [1, 1, 1])

        l.loc = [9.12,
                    4.1,
                    8.3]
        numpy.testing.assert_almost_equal(l.loc, [9.12, 4.1, 8.3], self.precs[1])

        l = LandmarkD()
        l.loc = [1,1,1]
        numpy.testing.assert_equal(l.loc, [1, 1, 1])

        l.loc = [9.12,
                    4.1,
                    8.3]
        numpy.testing.assert_almost_equal(l.loc, [9.12,
                                                    4.1,
                                                    8.3], self.precs[0])

    def test_get_scale(self):
        l = LandmarkF()
        nose.tools.assert_equal(l.scale, 1)

        l = LandmarkF([3, 4, 5], 44.5)
        nose.tools.assert_almost_equal(l.scale, 44.5, self.precs[1])

        l = LandmarkD()
        nose.tools.assert_equal(l.scale, 1)

        l = LandmarkD([3, 4, 5], 44.5)
        nose.tools.assert_almost_equal(l.scale, 44.5, self.precs[0])

    def test_set_scale(self):

        l = LandmarkF()
        l.scale = 1
        nose.tools.assert_equal(l.scale, 1)

        l.scale = 2
        nose.tools.assert_equal(l.scale, 2)

        l.scale = 2.456
        nose.tools.assert_almost_equal(l.scale, 2.456, self.precs[1])

        l.scale = -2
        nose.tools.assert_almost_equal(l.scale, -2, self.precs[1])

        l = LandmarkD()
        l.scale = 1
        nose.tools.assert_equal(l.scale, 1)

        l.scale = 2
        nose.tools.assert_equal(l.scale, 2)

        l.scale = 2.456
        nose.tools.assert_almost_equal(l.scale, 2.456, self.precs[0])

        l.scale = -2
        nose.tools.assert_almost_equal(l.scale, -2, self.precs[0])


    def test_normal(self):

        l = LandmarkF()
        numpy.testing.assert_equal(l.normal, [0,0,0])

        l.normal = [0,1,0]
        numpy.testing.assert_equal(l.normal, [0,1,0])

        l = LandmarkD()
        numpy.testing.assert_equal(l.normal, [0,0,0])

        l.normal = [0,1,0]
        numpy.testing.assert_equal(l.normal, [0,1,0])

    def test_covariance(self):
        covars = [Covar3d(7), Covar3f(7)]

        l = LandmarkF()

        # check default
        numpy.testing.assert_array_equal(l.covariance.matrix(), Covar3d().matrix())

        # set type-aligned covariance
        l.covariance = covars[1]
        numpy.testing.assert_array_almost_equal(l.covariance.matrix(), covars[1].matrix(), self.precs[1])

        l = LandmarkD()
        numpy.testing.assert_array_equal(l.covariance.matrix(), Covar3d().matrix())
        l.covariance = covars[0]
        numpy.testing.assert_array_almost_equal(l.covariance.matrix(), covars[0].matrix(), self.precs[0])

    def test_color(self):

        l = LandmarkF()

        # default
        nose.tools.assert_equal(l.color, RGBColor())
        c = RGBColor(0, 0, 0)
        l.color = c
        nose.tools.assert_equal(l.color, c)

        c = RGBColor(12, 240, 120)
        l.color = c
        nose.tools.assert_equal(l.color, c)

        l = LandmarkD()

        # default
        nose.tools.assert_equal(l.color, RGBColor())
        c = RGBColor(0, 0, 0)
        l.color = c
        nose.tools.assert_equal(l.color, c)

        c = RGBColor(12, 240, 120)
        l.color = c
        nose.tools.assert_equal(l.color, c)

    def test_observations(self):

        l = LandmarkF()

        # default
        nose.tools.assert_equal(l.observations, 0)

        l.observations = 42
        nose.tools.assert_equal(l.observations, 42)

        l = LandmarkD()

        # default
        nose.tools.assert_equal(l.observations, 0)

        l.observations = 42
        nose.tools.assert_equal(l.observations, 42)

    def test_cos_obs_angle(self):

        l = LandmarkF()
        # default
        nose.tools.assert_equal(l.cos_obs_angle, 1)
        l.cos_obs_angle = 0.5
        numpy.testing.assert_almost_equal(l.cos_obs_angle, 0.5, self.precs[1])

        l.cos_obs_angle = -0.5
        numpy.testing.assert_almost_equal(l.cos_obs_angle, -0.5, self.precs[1])
        # Can technically go outside range of cosine
        l.cos_obs_angle = 1.5
        numpy.testing.assert_almost_equal(l.cos_obs_angle, 1.5, self.precs[1])

        l.cos_obs_angle = -1.5
        numpy.testing.assert_almost_equal(l.cos_obs_angle, -1.5, self.precs[1])

        l = LandmarkD()
        # default
        nose.tools.assert_equal(l.cos_obs_angle, 1)
        l.cos_obs_angle = 0.5
        numpy.testing.assert_almost_equal(l.cos_obs_angle, 0.5, self.precs[0])

        l.cos_obs_angle = -0.5
        numpy.testing.assert_almost_equal(l.cos_obs_angle, -0.5, self.precs[0])
        # Can technically go outside range of cosine
        l.cos_obs_angle = 1.5
        numpy.testing.assert_almost_equal(l.cos_obs_angle, 1.5, self.precs[0])

        l.cos_obs_angle = -1.5
        numpy.testing.assert_almost_equal(l.cos_obs_angle, -1.5, self.precs[0])
