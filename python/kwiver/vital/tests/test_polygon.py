# """
# ckwg +31
# Copyright 2020 by Kitware, Inc.
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

# Tests for Python interface to vital::polygon

# """

from kwiver.vital.types import Polygon
import nose.tools as nt
import numpy as np


class TestVitalPolygon(object):
    def _create_points(self):
        return [
            np.array([10, 10]),
            np.array([10, 50]),
            np.array([50, 50]),
            np.array([30, 30]),
        ]

    def _create_polygons(self):
        pts = self._create_points()
        return (Polygon(), Polygon(pts))

    def test_new(self):
        pts = self._create_points()
        Polygon()
        Polygon(pts)

    def test_at_out_of_bounds(self):
        p1, p2 = self._create_polygons()
        nt.assert_raises(IndexError, p1.at, 0)
        nt.assert_raises(IndexError, p2.at, 4)

    def test_at_with_initial_points(self):
        pts = self._create_points()
        _, p2 = self._create_polygons()

        np.testing.assert_array_equal(pts[0], p2.at(0))
        np.testing.assert_array_equal(pts[1], p2.at(1))
        np.testing.assert_array_equal(pts[2], p2.at(2))
        np.testing.assert_array_equal(pts[3], p2.at(3))

        nt.assert_equal(p2.num_vertices(), 4)

    def test_at_point_not_in_array(self):
        _, p2 = self._create_polygons()
        nt.assert_raises(
            AssertionError, np.testing.assert_array_equal, np.array([17, 200]), p2.at(0)
        )
        nt.assert_raises(
            AssertionError, np.testing.assert_array_equal, np.array([17, 200]), p2.at(1)
        )
        nt.assert_raises(
            AssertionError, np.testing.assert_array_equal, np.array([17, 200]), p2.at(2)
        )
        nt.assert_raises(
            AssertionError, np.testing.assert_array_equal, np.array([17, 200]), p2.at(3)
        )

    def test_initial_num_vertices(self):
        p1, p2 = self._create_polygons()
        nt.assert_equal(p1.num_vertices(), 0)
        nt.assert_equal(p2.num_vertices(), 4)

    def test_push_back(self):
        p1, p2 = self._create_polygons()
        pts = self._create_points()
        # Start with p1
        # Check pushing back np_arrays
        p1.push_back(pts[0])
        nt.assert_equal(p1.num_vertices(), 1)

        p1.push_back(pts[1])
        nt.assert_equal(p1.num_vertices(), 2)

        # Check pushing back x and y coords
        p1.push_back(pts[2][0], pts[2][1])
        nt.assert_equal(p1.num_vertices(), 3)

        p1.push_back(pts[3][0], pts[3][1])
        nt.assert_equal(p1.num_vertices(), 4)

        np.testing.assert_array_equal(pts[0], p1.at(0))
        np.testing.assert_array_equal(pts[1], p1.at(1))
        np.testing.assert_array_equal(pts[2], p1.at(2))
        np.testing.assert_array_equal(pts[3], p1.at(3))

        # Now p2
        # Check np_arrays
        temp_pt = np.array([40, 35])
        p2.push_back(temp_pt)
        nt.assert_equal(p2.num_vertices(), 5)

        # Check x and y coords
        x, y = 20, 10
        p2.push_back(x, y)
        nt.assert_equal(p2.num_vertices(), 6)

        np.testing.assert_array_equal(temp_pt, p2.at(4))
        np.testing.assert_array_equal(np.array([x, y]), p2.at(5))

    def test_contains(self):
        pts = self._create_points()
        p1, p2 = self._create_polygons()

        # p1 shouldn't contain any points
        nt.assert_false(p1.contains(pts[0]))  # Check np arrays
        nt.assert_false(p1.contains(pts[1]))  # Check np arrays
        nt.assert_false(p1.contains(pts[2][0], pts[2][1]))  # Check x and y coords
        nt.assert_false(p1.contains(pts[3][0], pts[3][1]))  # Check x and y coords

        # p2 should contain all, as they are vertices
        nt.ok_(p2.contains(pts[0]))  # Check np arrays
        nt.ok_(p2.contains(pts[1]))  # Check np arrays
        nt.ok_(p2.contains(pts[2][0], pts[2][1]))  # Check x and y coords
        nt.ok_(p2.contains(pts[3][0], pts[3][1]))  # Check x and y coords

        # p2 should also contain boundary points,
        # and points inside the shape
        nt.ok_(p2.contains(25, 40))  # x and y coord inside
        nt.ok_(p2.contains(35, 50))  # x and y coord boundary
        nt.ok_(p2.contains(np.array([20, 30])))  # np_array inside
        nt.ok_(p2.contains(np.array([10, 30])))  # np_array boundary

    def test_get_vertices(self):
        pts = self._create_points()
        p1, p2 = self._create_polygons()

        # p1
        np.testing.assert_array_equal(p1.get_vertices(), [])
        p1.push_back(pts[0])
        p1.push_back(pts[1])
        p1.push_back(pts[2])
        p1.push_back(pts[3])
        np.testing.assert_array_equal(p1.get_vertices(), pts)

        # p2
        np.testing.assert_array_equal(p2.get_vertices(), pts)
        temp_pt = np.array([40, 35])
        p2.push_back(temp_pt)
        np.testing.assert_array_equal(p2.get_vertices(), pts + [temp_pt])
