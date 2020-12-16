"""
ckwg +31
Copyright 2016 by Kitware, Inc.
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

Tests for class Feature, interfacing vital::feature functionality

"""
import random
import unittest

import nose.tools
import numpy

from kwiver.vital.types import Covariance, EigenArray, Feature, RGBColor


class TestFeature (unittest.TestCase):

    def test_new(self):
        f1 = Feature()
        f2 = Feature([1, 1], 1, 2, 1)

    def test_get_typename(self):
        # Returns C++ std::type_info.name values
        f = Feature(ctype='d')
        nose.tools.assert_equal(f.type_name, 'd')

        f = Feature(ctype='f')
        nose.tools.assert_equal(f.type_name, 'f')

    def test_get_location(self):
        f = Feature()
        numpy.testing.assert_almost_equal(
            f.location,
            [0, 0]
        )

        expected = [12.3, 643]
        f = Feature(loc=expected)
        numpy.testing.assert_almost_equal(
            f.location,
            expected
        )
        # iterable form
        f = Feature(loc=(12.3, 643))
        numpy.testing.assert_almost_equal(
            f.location,
            expected
        )

    def test_get_mag(self):
        f = Feature()
        nose.tools.assert_equal(f.magnitude, 0)

        f = Feature(mag=1.1)
        nose.tools.assert_equal(f.magnitude, 1.1)

    def test_get_scale(self):
        f = Feature()
        nose.tools.assert_equal(f.scale, 1)

        f = Feature(scale=2.1)
        nose.tools.assert_equal(f.scale, 2.1)

    def test_get_angle(self):
        f = Feature()
        nose.tools.assert_equal(f.angle, 0)

        f = Feature(angle=1.1)
        nose.tools.assert_equal(f.angle, 1.1)

    def test_get_covar(self):
        dflt_covar = Covariance.new_covar(2)
        f = Feature()
        #nose.tools.assert_equal(f.covariance, dflt_covar)
        # No constructor slot to initialize non-default covariance

    def test_get_color(self):
        dflt_color = RGBColor()
        f = Feature()
        nose.tools.assert_equal(f.color, dflt_color)

        c = RGBColor(5, 32, 10)
        f = Feature(rgb_color=c)
        nose.tools.assert_equal(f.color, c)

    def test_set_location(self):
        f = Feature(ctype='d')
        expected = [random.random(),
                    random.random()]
        f.location = expected
        # making sure that we went through the setter, and not just setting the
        # exact value to the property
        numpy.testing.assert_almost_equal(f.location, expected, 16)

        f = Feature(ctype='f')
        expected = [random.random(),
                    random.random()]
        f.location = expected
        numpy.testing.assert_almost_equal(f.location, expected, 6)

    def test_set_magnitude(self):
        f = Feature(ctype='d')
        nose.tools.assert_equal(f.magnitude, 0)  # default value
        expected = random.random()
        f.magnitude = expected
        nose.tools.assert_almost_equal(f.magnitude, expected, 16)

        f = Feature(ctype='f')
        nose.tools.assert_equal(f.magnitude, 0)  # default value
        expected = random.random()
        f.magnitude = expected
        nose.tools.assert_almost_equal(f.magnitude, expected, 6)

    def test_set_scale(self):
        f = Feature(ctype='d')
        nose.tools.assert_equal(f.scale, 1)  # default value
        expected = random.random()
        f.scale = expected
        nose.tools.assert_almost_equal(f.scale, expected, 16)

        f = Feature(ctype='f')
        nose.tools.assert_equal(f.scale, 1)  # default value
        expected = random.random()
        f.scale = expected
        nose.tools.assert_almost_equal(f.scale, expected, 6)

    def test_set_angle(self):
        f = Feature(ctype='d')
        nose.tools.assert_equal(f.angle, 0)  # default value
        expected = random.random()
        f.angle = expected
        nose.tools.assert_almost_equal(f.angle, expected, 16)

        f = Feature(ctype='f')
        nose.tools.assert_equal(f.angle, 0)  # default value
        expected = random.random()
        f.angle = expected
        nose.tools.assert_almost_equal(f.angle, expected, 6)

    def test_set_covar(self):
        f = Feature(ctype='d')
        #nose.tools.assert_equal(f.covariance, Covariance.new_covar())

        expected = [[1, 2],
                    [3, 4]]
        c = Covariance.from_matrix(2, 'd', expected)
        f.covariance = c
        #nose.tools.assert_equal(f.covariance, c)
        # Should also work if we just give it the raw iterable
        f.covariance = expected
        #nose.tools.assert_equal(f.covariance, c)

        # And for floats...
        f = Feature(ctype='f')
        #nose.tools.assert_equal(f.covariance, Covariance())

        expected = [[1, 2],
                    [3, 4]]
        c = Covariance.from_matrix(2, 'f', expected)
        f.covariance = c
        #nose.tools.assert_equal(f.covariance, c)
        # Should also work if we just give it the raw iterable
        f.covariance = expected
        #nose.tools.assert_equal(f.covariance, c)

    def test_set_color(self):
        expected = RGBColor(4, 20, 0)

        f = Feature(ctype='d')
        nose.tools.assert_equal(f.color, RGBColor())
        f.color = expected
        nose.tools.assert_equal(f.color, expected)

        f = Feature(ctype='f')
        nose.tools.assert_equal(f.color, RGBColor())
        f.color = expected
        nose.tools.assert_equal(f.color, expected)
