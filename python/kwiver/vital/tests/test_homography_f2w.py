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

Tests for python F2WHomography interface

"""

import nose.tools as nt
import numpy as np

from kwiver.vital.types import F2WHomography, HomographyD, HomographyF


class TestF2WHomography(object):
    def test_frame_init(self):
        F2WHomography(5)
        F2WHomography(-7)
        F2WHomography(0)

    def test_homography_init(self):
        hd = HomographyD()
        hf = HomographyF()

        F2WHomography(hd, 5)
        F2WHomography(hf, 5)

        hd = HomographyD.random()
        hf = HomographyF.random()

        F2WHomography(hd, 5)
        F2WHomography(hf, 5)

    def test_copy_construct(self):
        f2w = F2WHomography(10)
        F2WHomography(f2w)

        f2w = F2WHomography(HomographyD.random(), -5)
        F2WHomography(f2w)

    def test_get_homography(self):
        homs = [
            HomographyD(),
            HomographyF(),
            HomographyD.random(),
            HomographyF.random(),
        ]
        for h in homs:
            f2w = F2WHomography(h, 5)
            np.testing.assert_array_equal(f2w.homography.matrix(), h.matrix())

            f2w_copy = F2WHomography(f2w)
            np.testing.assert_array_equal(f2w_copy.homography.matrix(), h.matrix())

    def test_get_frame_id(self):
        f_ids = [5, 0, -10]
        for f_id in f_ids:
            f2w = F2WHomography(f_id)
            nt.assert_equal(f2w.frame_id, f_id)

            f2w = F2WHomography(HomographyD(), f_id)
            nt.assert_equal(f2w.frame_id, f_id)

            f2w_copy = F2WHomography(f2w)
            nt.assert_equal(f2w_copy.frame_id, f_id)

    def check_each_element_equal(self, f2w, hom):
        mat = hom.matrix()
        for i in range(3):
            for j in range(3):
                nt.assert_almost_equal(f2w.get(i, j), mat[i, j])

    def test_get(self):
        homs = [HomographyD.random(), HomographyF.random()]
        for hom in homs:
            f2w = F2WHomography(5)
            self.check_each_element_equal(f2w, HomographyD())

            f2w = F2WHomography(hom, 0)
            self.check_each_element_equal(f2w, hom)

            f2w_copy = F2WHomography(f2w)
            self.check_each_element_equal(f2w_copy, hom)

    def test_get_oob(self):
        f2ws = F2WHomography(5), F2WHomography(HomographyD.random(), 5)
        exp_err_msg = "Tried to perform get\\(\\) out of bounds"
        for f2w in f2ws:
            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2w.get(3, 0)

            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2w.get(-4, 0)

            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2w.get(0, 3)

            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2w.get(0, -4)

            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2w.get(5, 5)

            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2w.get(-6, -6)
