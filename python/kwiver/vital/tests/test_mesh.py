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

import unittest
from pathlib import Path

from kwiver.vital.types import Mesh
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parents[0] / "data"


class TestMesh(unittest.TestCase):
    @classmethod
    def setUp(self):
        pass

    def test_constructor(self):
        Mesh()
        Mesh.from_ply_file(str(TEST_DATA_DIR) + "/cube.ply")

    def test_object_properties(self):
        m = Mesh.from_ply_file(str(TEST_DATA_DIR) + "/cube.ply")
        self.assertTrue(m.is_init())
        self.assertEqual(m.num_verts(), 8)
        self.assertEqual(m.num_faces(), 6)
        self.assertEqual(m.num_edges(), 0)

    def test_bad_mesh(self):
        m = Mesh()
        self.assertTrue(not m.is_init())
        self.assertEqual(m.num_verts(), 0)
        self.assertEqual(m.num_faces(), 0)
        self.assertEqual(m.num_edges(), 0)
