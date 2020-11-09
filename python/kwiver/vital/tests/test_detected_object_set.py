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

Tests for DetectedObjectSet interface class.

"""
import unittest
import nose.tools
import numpy as np
from kwiver.vital.tests.cpp_helpers import det_obj_set_helpers as dos_helper
from kwiver.vital.types import (
    DetectedObjectSet as dos,
    BoundingBoxD as bb,
    descriptor,
    DetectedObject as do,
    DetectedObjectType as dot,
    Image,
    ImageContainer,
    geodesy,
    GeoPoint,
    Point2d
)
class SimpleDetectedSet(dos):
    def __init__(self,do_):
        dos.__init__(self)
        self.det_objs = do_
    def size(self):
        return 1
    def empty(self):
        return False
    def at(self,loc):
        return do(bb(10, 10, 20, 20))


class TestDetectedObjectSet(unittest.TestCase):

    def setUp(self):

        # Values to setup Detected Object to hold in DOS
        self.bbox1 = bb(10, 10, 20, 20)
        self.bbox2 = bb(10, 10, 30, 30)
        self.bbox3 = bb(5, 5, 20, 20)
        self.bbox4 = bb(1, 1, 10, 10)
        self.conf = 0.5
        self.conf2 = 0.4
        self.conf3 = 0.75
        self.cm = dot("example_class", 0.4)
        self.cm2 = dot("foo2", 3.14)
        self.cm3 = dot("foo3", 0.11)
        self.mask = ImageContainer(Image(1080, 720))
        self.mask2 = ImageContainer(Image(1920, 1080))
        self.mask3 = ImageContainer(Image(720, 1080))

        # Establish set of DO objects to pass to DOS constructor
        self.set = np.array([do(self.bbox1, self.conf, self.cm, self.mask),
                             do(self.bbox2, self.conf2, self.cm2, self.mask2),
                             do(self.bbox3, self.conf3, self.cm3, self.mask3),
                             do(self.bbox4)])

    def test_Constructor(self):
        dos()
        dos(self.set)

    def test_virtualOverrides(self):
        # Test __init__
        do_ = do(self.bbox4)
        d = SimpleDetectedSet(do_)

        # Test inheritance
        self.assertTrue(issubclass(SimpleDetectedSet, dos))

        # Test size override
        size_over = dos_helper.call_size(d)
        self.assertEqual(size_over, 1)

        # Test empty override
        empty_over = dos_helper.call_empty(d)
        self.assertFalse(empty_over)

        # # Test at override
        at_over = dos_helper.call_at(d, 0)
        self.assertIsInstance(at_over, do)

    def test_MFunctions(self):
        d = dos(self.set)
        # Test Empty/Size
        d_empt = dos()
        self.assertFalse(d.empty())
        self.assertEqual(d.size(), 4)
        self.assertTrue(d_empt.empty())
        self.assertEqual(d_empt.size(), 0)

        # Test Add
        tst_do = do(self.bbox1, self.conf2, self.cm3, self.mask)
        tst_dos = dos()
        d.add(tst_do)
        self.assertEqual(d.size(), 5)
        d.add(tst_dos)
        self.assertEqual(d.size(), 5)
        tst_dos_full = dos(np.array([do(self.bbox1)]))
        d.add(tst_dos_full)
        self.assertEqual(d.size(), 6)

        # Test Select
        sel = d.select()
        self.assertEqual(len(sel), 6)
        sel = d.select(0.41)
        self.assertEqual(len(sel), 4)
        sel = d.select(class_name = "foo2")
        self.assertEqual(len(sel), 1 )
        sel = d.select(0, "foo3")
        self.assertEqual(len(sel), 2)
        sel = d.select(5.0, "example_class")
        self.assertEqual(len(sel), 0)
        self.assertIsInstance(sel, dos)
        sel = d.select(class_name = "foo3")
        do_at = sel.at(0)
        self.assertIsInstance(do_at, do)
        self.assertEqual(do_at.__nice__(), "conf=0.75")

        # Test Clone
        d_clone = d.clone()
        self.assertIsInstance(d_clone, dos)
        self.assertEqual(d_clone.size(), 6)
        self.assertFalse(d_clone.empty())
        sel_clone = d_clone.select()
        self.assertEqual(len(sel_clone), 6)
        sel_clone = d_clone.select(class_name = "foo3")
        self.assertEqual(len(sel_clone),2)
        do_at_clone = sel_clone.at(0)
        self.assertIsInstance(do_at_clone, do)
        self.assertEqual(do_at_clone.__nice__(), "conf=0.75")

    def test_PyFunc(self):
        t = dos(self.set)
        self.assertIsInstance(str(t), str)
        self.assertEqual(str(t), "<DetectedObjectSet(size=4)>")
        self.assertIsInstance(t.__repr__(), str)
        self.assertIsInstance(t.__nice__(), str)
        self.assertEqual(t.__nice__(), "size=4")
        self.assertEqual(t.__getitem__(1),t.at(1))
        itr = iter(t)
        self.assertIsInstance(next(itr),do)
