# """
# ckwg +31
# Copyright 2016-2020 by Kitware, Inc.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ==============================================================================

# Tests for class Feature, interfacing vital::feature functionality

# """
import random
import unittest

import nose.tools as nt
import numpy as np

from kwiver.vital.types import Covar2d, Covar2f, Feature, FeatureF, FeatureD, RGBColor


class TestVitalFeature(unittest.TestCase):
    def test_new(self):
        f1 = FeatureF([1, 1], 1, 2, 1)
        f2 = FeatureF()

        f3 = FeatureD([1, 1], 1, 2, 1)
        f4 = FeatureD()

    def test_no_construct_base(self):
        with nt.assert_raises_regexp(
            TypeError, "kwiver.vital.types.feature.Feature: No constructor defined!"
        ):
            Feature()

    def test_get_typename(self):
        f = FeatureD()
        nt.assert_equal(f.type_name, "d")

        f = FeatureF()
        nt.assert_equal(f.type_name, "f")

    def test_get_default_location(self):
        f = FeatureD()
        np.testing.assert_almost_equal(f.location, [0, 0])

        expected = [12.3, 643]
        f = FeatureD(loc=expected)
        np.testing.assert_almost_equal(f.location, expected)
        # iterable form
        f = FeatureD(loc=(12.3, 643))
        np.testing.assert_almost_equal(f.location, expected)

    def test_get_default_mag(self):
        f = FeatureD()
        nt.assert_equal(f.magnitude, 0)

        f = FeatureD([1, 1], mag=1.1)
        nt.assert_equal(f.magnitude, 1.1)

    def test_get_default_scale(self):
        f = FeatureD()
        nt.assert_equal(f.scale, 1)

        f = FeatureD([1, 1], scale=2.1)
        nt.assert_equal(f.scale, 2.1)

    def test_get_default_angle(self):
        f = FeatureD()
        nt.assert_equal(f.angle, 0)

        f = FeatureD([1, 1], angle=1.1)
        nt.assert_equal(f.angle, 1.1)

    def test_get_default_covar(self):
        dflt_covar = Covar2d()
        f = FeatureD()
        np.testing.assert_array_equal(f.covariance.matrix(), dflt_covar.matrix())
        # No constructor slot to initialize non-default covariance

    def test_get_default_color(self):
        dflt_color = RGBColor()
        f = FeatureD()
        nt.assert_equal(f.color, dflt_color)

        c = RGBColor(5, 32, 10)
        f = FeatureD([1, 1], rgb_color=c)
        nt.assert_equal(f.color, c)

    def test_set_and_get_location(self):
        f = FeatureD()
        expected = [random.random(), random.random()]
        f.location = expected
        # making sure that we went through the setter, and not just setting the
        # exact value to the property
        np.testing.assert_array_almost_equal(f.location, expected, 16)

        f = FeatureF()
        expected = [random.random(), random.random()]
        f.location = expected
        np.testing.assert_array_almost_equal(f.location, expected, 6)

    def test_set_and_get_magnitude(self):
        f = FeatureD()
        expected = random.random()
        f.magnitude = expected
        nt.assert_almost_equal(f.magnitude, expected, 16)

        f = FeatureF()
        expected = random.random()
        f.magnitude = expected
        nt.assert_almost_equal(f.magnitude, expected, 6)

    def test_set_and_get_scale(self):
        f = FeatureD()
        expected = random.random()
        f.scale = expected
        nt.assert_almost_equal(f.scale, expected, 16)

        f = FeatureF()
        expected = random.random()
        f.scale = expected
        nt.assert_almost_equal(f.scale, expected, 6)

    def test_set_and_get_angle(self):
        f = FeatureD()
        expected = random.random()
        f.angle = expected
        nt.assert_almost_equal(f.angle, expected, 16)

        f = FeatureF()
        expected = random.random()
        f.angle = expected
        nt.assert_almost_equal(f.angle, expected, 6)

    def test_set_and_get_covar(self):
        f = FeatureD()

        expected = [[1, 2], [3, 4]]
        c = Covar2d(expected)
        f.covariance = c
        np.testing.assert_array_almost_equal(f.covariance.matrix(), c.matrix(), 16)

        # And for floats...
        f = FeatureF()

        c = Covar2f(expected)
        f.covariance = c
        np.testing.assert_array_almost_equal(f.covariance.matrix(), c.matrix(), 6)

    def test_set_and_get_color(self):
        expected = RGBColor(4, 20, 0)

        f = FeatureD()
        f.color = expected
        nt.assert_equal(f.color, expected)

        f = FeatureF()
        f.color = expected
        nt.assert_equal(f.color, expected)

    def comparison_helper(self, f1, f2, eq_=False, eq_except_angle=False):
        nt.assert_equals(f1 == f2, eq_)
        nt.assert_equals(f1 != f2, not eq_)
        nt.assert_equals(f1.equal_except_for_angle(f2), eq_except_angle)

    def test_comparisons(self):
        nt.assert_false(FeatureD() == FeatureF())

        ctors = [FeatureD, FeatureF]
        for ctor in ctors:
            f1, f2 = ctor(), ctor()
            self.comparison_helper(f1, f2, True, True)

            # location
            f2.location = f1.location + 1
            self.comparison_helper(f1, f2)
            f2.location = f1.location
            self.comparison_helper(f1, f2, True, True)

            # magnitude
            f2.magnitude = f1.magnitude + 1
            self.comparison_helper(f1, f2)
            f2.magnitude = f1.magnitude
            self.comparison_helper(f1, f2, True, True)

            # scale
            f2.scale = f1.scale + 1
            self.comparison_helper(f1, f2)
            f2.scale = f1.scale
            self.comparison_helper(f1, f2, True, True)

            # angle
            f2.angle = f1.angle + 1
            self.comparison_helper(f1, f2, eq_except_angle=True)
            f2.angle = f1.angle
            self.comparison_helper(f1, f2, True, True)

            # covariance
            f2.covariance[0, 0] = f1.covariance[0, 0] + 1
            self.comparison_helper(f1, f2)
            f2.covariance[0, 0] = f1.covariance[0, 0]
            self.comparison_helper(f1, f2, True, True)

            # color
            f2.color = RGBColor(r=f1.color.r + 1)
            self.comparison_helper(f1, f2)
            f2.color = f1.color
            self.comparison_helper(f1, f2, True, True)

            # Try many at once
            f2.location = f1.location + 1
            self.comparison_helper(f1, f2)

            f2.magnitude = f1.magnitude + 1
            self.comparison_helper(f1, f2)

            f2.scale = f1.scale + 1
            self.comparison_helper(f1, f2)

            f2.angle = f1.angle + 1
            self.comparison_helper(f1, f2)

            f2.covariance[0, 0] = f1.covariance[0, 0] + 1
            self.comparison_helper(f1, f2)

            f2.color = RGBColor(r=f1.color.r + 1)
            self.comparison_helper(f1, f2)

    def test_copy_constructor(self):
        double_def = FeatureD()
        double_nondef = FeatureD([1, 1], 1, 2, 1)

        float_def = FeatureF()
        float_nondef = FeatureF([1, 1], 1, 2, 1)

        nt.ok_(FeatureD(double_def)    == double_def)
        nt.ok_(FeatureD(double_nondef) == double_nondef)
        nt.ok_(FeatureF(double_def)    != double_def)
        nt.ok_(FeatureF(double_nondef) != double_nondef)

        nt.ok_(FeatureF(float_def)    == float_def)
        nt.ok_(FeatureF(float_nondef) == float_nondef)
        nt.ok_(FeatureD(float_def)    != float_def)
        nt.ok_(FeatureD(float_nondef) != float_nondef)

    def test_clone(self):
        features = [
            FeatureD(),
            FeatureD([1, 1], 1, 2, 1),
            FeatureF(),
            FeatureF([1, 1], 1, 2, 1),
        ]
        for f in features:
            f_clone = f.clone()
            nt.ok_(f == f_clone)

            # Changing one doesn't reflect in the other
            f_clone.scale = f.scale + 1
            nt.ok_(f != f_clone)

    def test_to_str_default(self):
        features = [FeatureD(), FeatureF()]
        for f in features:
            s = str(f).split()

            nt.assert_equals(s[0], "0")  # loc[0]
            nt.assert_equals(s[1], "0")  # loc[1]
            nt.assert_equals(s[2], "0")  # mag
            nt.assert_equals(s[3], "1")  # scale
            nt.assert_equals(s[4], "0")  # angle
            nt.assert_equals(s[5], "255")  # color.r
            nt.assert_equals(s[6], "255")  # color.g
            nt.assert_equals(s[7], "255")  # color.b

        print("Default feature representation:", features[0], sep="\n")

    def test_to_str(self):
        expected_loc = np.array([random.random(), random.random()])
        expected_color = RGBColor(50, 60, 70)
        features = [
            FeatureD(expected_loc, 1.5, 2.5, 1.7, expected_color),
            FeatureF(expected_loc, 1.5, 2.5, 1.7, expected_color),
        ]

        for f in features:
            s = str(f).split()

            # Notice 6 is used here, as that is the
            # default precision used for C++ stringstreams
            np.testing.assert_almost_equal(float(s[0]), expected_loc[0], 6)  # loc[0]
            np.testing.assert_almost_equal(float(s[1]), expected_loc[1], 6)  # loc[1]
            np.testing.assert_almost_equal(float(s[2]), 1.5)  # mag
            np.testing.assert_almost_equal(float(s[3]), 2.5)  # scale
            np.testing.assert_almost_equal(float(s[4]), 1.7)  # angle
            nt.assert_equals(int(s[5]), expected_color.r)  # color.r
            nt.assert_equals(int(s[6]), expected_color.g)  # color.g
            nt.assert_equals(int(s[7]), expected_color.b)  # color.b

        print("Non default feature representation:", features[0], sep="\n")
