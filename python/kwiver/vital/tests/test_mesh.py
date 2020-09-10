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

Tests for the vital class mesh

"""

import numpy as np
import numpy.testing as npt
import unittest
import nose.tools as nt

from kwiver.vital.types import Mesh


class TestMesh(unittest.TestCase):
    @classmethod
    def setUp(self):
        pass
    def test_constructor(self):
        Mesh()
        Mesh.from_ply_file("data/airplane.ply")

    def test_object_properties(self):
        m = Mesh.from_ply_file("data/airplane.ply")
        nt.ok_(m.is_init())
        nt.assert_equal(m.num_verts(), 1335)
        nt.assert_equal(m.num_faces(), 2452)
        nt.assert_equal(m.num_edges(), 0)

    def test_bad_mesh(self):
        m = Mesh()
        nt.ok_(not m.is_init())
        nt.assert_equal(m.num_verts(), 0)
        nt.assert_equal(m.num_faces(), 0)
        nt.assert_equal(m.num_edges(), 0)
