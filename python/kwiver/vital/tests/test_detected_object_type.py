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


Tests for DetectedObjectType interface class.

"""
import unittest
import nose.tools
import numpy as np


from kwiver.vital.types import DetectedObjectType as DOT


class TestDetectedObject(unittest.TestCase):

    def test_constructor(self):
        DOT()
        DOT(np.array(["name1","class_name2","class3"]),np.array([1.0,2.3,3.14]))
        DOT("name", 2.0)

    def test_methods(self):
        t = DOT(np.array(["name1","class_name2","class3"]),np.array([1.0,2.3,3.14]))

        # str/repr/itr
        self.assertIsInstance(str(t),str)
        self.assertIsInstance(t.__repr__(), str)
        itr = iter(t)
        self.assertIsInstance(next(itr),tuple)

        # Has Class Name
        self.assertTrue(t.has_class_name("name1"))
        self.assertFalse(t.has_class_name("foo"))

        # Score
        self.assertEqual(t.score("class_name2"),2.3)

        # Get most likely
        self.assertEqual(t.get_most_likely_class(),"class3")
        self.assertEqual(3.14,t.get_most_likely_score())

        # Set Score
        t.set_score("foo1",3.8)
        self.assertEqual(3.8,t.score("foo1"))
        t.set_score("foo2",3.9)
        self.assertTrue(t.has_class_name("foo2"))
        self.assertEqual(t.score("foo2"),3.9)

        # Delete Score
        t.delete_score("foo1")
        self.assertFalse(t.has_class_name("foo1"))

        # Class Names
        np.testing.assert_array_equal(t.class_names(), np.array(["foo2","class3","class_name2","name1"]))
        np.testing.assert_array_equal(t.class_names(3.0), np.array(["foo2","class3"]))

        np.testing.assert_array_equal(t.all_class_names(),np.array(["class3","class_name2","foo1","foo2","name","name1"]))
        print()
        print("--------------------------")
        print(t.all_class_names())
        print("--------------------------")
        for item in t.all_class_names():
            print()
            print("--------------")
            print(item)
            print("--------------")
