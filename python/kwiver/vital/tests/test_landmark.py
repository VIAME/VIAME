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

Tests for Landmark interface

"""
from __future__ import print_function
import unittest

import nose.tools
import numpy

from kwiver.vital.types import Landmark, Covariance, RGBColor


class TestLandmark (unittest.TestCase):

    ctypeS = ['d', 'f']

    def test_new(self):
        Landmark()
        Landmark(ctype='f')

        Landmark([1, 1, 1])
        Landmark([1, 1, 1], ctype='f')

        Landmark(scale=10)
        Landmark(scale=10, ctype='f')

    def test_type_name(self):
        # Double default
        nose.tools.assert_equal(Landmark().type_name, 'd')
        nose.tools.assert_equal(Landmark(ctype='f').type_name, 'f')

        nose.tools.assert_equal(Landmark([1, 2, 2]).type_name, 'd')
        nose.tools.assert_equal(Landmark([1, 2, 2], ctype='f').type_name, 'f')

    def test_get_loc(self):
        for ct in self.ctypeS:
            print(ct)

            l = Landmark(ctype=ct)
            numpy.testing.assert_equal(l.loc, [0,0,0])

            l = Landmark([1, 2, 3], ctype=ct)
            numpy.testing.assert_equal(l.loc, [1,2,3])

    def test_set_loc(self):
        for ct in self.ctypeS:
            print(ct)

            l = Landmark(ctype=ct)
            l.loc = [1,1,1]
            numpy.testing.assert_equal(l.loc, [1, 1, 1])

            l.loc = [9.12,
                     4.1,
                     8.3]
            numpy.testing.assert_almost_equal(l.loc, [9.12,
                                                      4.1,
                                                      8.3], 6)

    def test_get_scale(self):
        for ct in self.ctypeS:
            l = Landmark(ctype=ct)
            print(ct)

            nose.tools.assert_equal(l.scale, 1)

            l = Landmark(scale=17, ctype=ct)
            nose.tools.assert_equal(l.scale, 17)

            l = Landmark(scale=2.22, ctype=ct)
            nose.tools.assert_almost_equal(l.scale, 2.22, 6)

            l = Landmark([3, 4, 5], 44.5, ct)
            nose.tools.assert_almost_equal(l.scale, 44.5, 6)

    def test_set_scale(self):
        for ct in self.ctypeS:
            print(ct)

            l = Landmark(ctype=ct)
            l.scale = 1
            nose.tools.assert_equal(l.scale, 1)

            l.scale = 2
            nose.tools.assert_equal(l.scale, 2)

            l.scale = 2.456
            nose.tools.assert_almost_equal(l.scale, 2.456, 6)

            l.scale = -2
            nose.tools.assert_almost_equal(l.scale, -2, 6)


    def test_normal(self):
        for ct in self.ctypeS:
            print(ct)
            l = Landmark(ctype=ct)

            # check default
            numpy.testing.assert_equal(l.normal, [0,0,0])

            l.normal = [0,1,0]
            numpy.testing.assert_equal(l.normal, [0,1,0])

    def test_covariance(self):
        for ct in self.ctypeS:
            print(ct)
            l = Landmark(ctype=ct)

            # check default
            #nose.tools.assert_equal(l.covariance, Covariance.new_covar(3))

            # set type-aligned covariance
            c = Covariance.new_covar(3, ct, 7)
            l.covariance = c
            #nose.tools.assert_equal(l.covariance, c)

    def test_color(self):
        for ct in self.ctypeS:
            print(ct)
            l = Landmark(ctype=ct)

            # default
            nose.tools.assert_equal(l.color, RGBColor())

            c = RGBColor(0, 0, 0)
            l.color = c
            nose.tools.assert_equal(l.color, c)

            c = RGBColor(12, 240, 120)
            l.color = c
            nose.tools.assert_equal(l.color, c)

    def test_observations(self):
        for ct in self.ctypeS:
            print(ct)
            l = Landmark(ctype=ct)

            # default
            nose.tools.assert_equal(l.observations, 0)

            l.observations = 42
            nose.tools.assert_equal(l.observations, 42)
