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

Tests for Python interface to vital::uid

"""

from kwiver.vital.types import UID
import unittest


class TestVitalUID(unittest.TestCase):
    # Creates some uid objects we'll use for testing
    # Note that if a uid represents a valid string, its name is the string
    # Otherwise, it's empty, and is named as such
    def _create_uids(self):
        return (
            UID(),
            UID(""),
            UID("foo_bar", 0),  # copy 0 bytes
            UID("foo_bar", 4),  # copy 4 bytes
            UID("foo_bar"),  # copy entire string
            UID("baz_qux", 7),  # copy all 7 bytes
        )

    def test_new(self):
        uid_instance = UID()
        uid_instance = UID("")
        uid_instance = UID("foo_bar", 0)
        uid_instance = UID("foo_bar", 4)
        uid_instance = UID("foo_bar")
        uid_instance = UID("baz_qux", 7)

    def test_is_valid(self):
        (empty1, empty2, empty3, foo_, foo_bar, baz_qux) = self._create_uids()
        # 1, 2, 4 should not be valid
        self.assertTrue(
            not empty1.is_valid(), "uid with default constructor should not be valid"
        )
        self.assertTrue(
            not empty2.is_valid(), "uid with empty string should not be valid"
        )
        self.assertTrue(
            not empty3.is_valid(), "uid with 0 bytes copied should not be valid"
        )
        # 3, 5, 6 should be valid
        self.assertTrue(foo_.is_valid(), "uid with 4 bytes copied should be valid")
        self.assertTrue(foo_bar.is_valid(), "uid with non empty string copied be valid")
        self.assertTrue(baz_qux.is_valid(), "uid with all bytes copied should be valid")

    def test_value(self):
        (empty1, empty2, empty3, foo_, foo_bar, baz_qux) = self._create_uids()

        self.assertEqual(empty1.value(), "")
        self.assertEqual(empty2.value(), "")
        self.assertEqual(empty3.value(), "")
        self.assertEqual(foo_.value(), "foo_")
        self.assertEqual(foo_bar.value(), "foo_bar")
        self.assertEqual(baz_qux.value(), "baz_qux")

    def test_size_and_length(self):
        (empty1, empty2, empty3, foo_, foo_bar, baz_qux) = self._create_uids()

        self.assertEqual(empty1.size(), 0)
        self.assertEqual(empty2.size(), 0)
        self.assertEqual(empty3.size(), 0)
        self.assertEqual(foo_.size(), 4)
        self.assertEqual(foo_bar.size(), 7)
        self.assertEqual(baz_qux.size(), 7)

        self.assertEqual(len(empty1), empty1.size())
        self.assertEqual(len(empty2), empty2.size())
        self.assertEqual(len(empty3), empty3.size())
        self.assertEqual(len(foo_), foo_.size())
        self.assertEqual(len(foo_bar), foo_bar.size())
        self.assertEqual(len(baz_qux), baz_qux.size())

    def test_copy_equal(self):
        instances = self._create_uids()
        instances_copy = self._create_uids()

        for i in range(len(instances)):
            uid = instances[i]
            uid_copy = instances_copy[i]
            self.assertEqual(
                uid,
                uid_copy,
                "{} and {} uids not equal".format(uid.value(), uid_copy.value()),
            )

    def test_empty_equal(self):
        (empty1, empty2, empty3, foo_, _, _) = self._create_uids()
        self.assertEqual(empty1, empty2)
        self.assertEqual(empty1, empty3)
        self.assertEqual(empty2, empty3)

        self.assertTrue(not empty1 == foo_)
        self.assertTrue(not empty2 == foo_)
        self.assertTrue(not empty3 == foo_)

    # This creates some uid objects we'll use for the equals
    # and not equals tests. These are better tests than the objects
    # provided by _create_uids
    def _create_uids_for_equals_check(self):
        return (
            UID("test_str"),
            UID("test_str_abcde", 8),
            UID("test_str", 8),
            UID("test_strtest_str"),
            UID("Atest_str"),
            UID("test_strA"),
            UID("Test_str"),
        )

    def test_equal(self):
        (
            test_str1,
            test_str2,
            test_str3,
            test_strtest_str,
            Atest_str,
            test_strA,
            Test_str,
        ) = self._create_uids_for_equals_check()
        # Test that same string constructed different ways is equivalent
        self.assertEqual(test_str1, test_str2)
        self.assertEqual(test_str1, test_str3)
        self.assertEqual(test_str2, test_str3)

        self.assertTrue(not test_str1 == test_strtest_str)
        self.assertTrue(not test_str1 == Atest_str)
        self.assertTrue(not test_str1 == test_strA)
        self.assertTrue(not test_str1 == Test_str)

    def test_not_equals(self):
        (
            test_str1,
            test_str2,
            test_str3,
            test_strtest_str,
            Atest_str,
            test_strA,
            Test_str,
        ) = self._create_uids_for_equals_check()

        self.assertTrue(not test_str1 != test_str2)
        self.assertTrue(not test_str1 != test_str3)
        self.assertTrue(not test_str2 != test_str3)

        self.assertTrue(test_str1 != test_strtest_str)
        self.assertTrue(test_str1 != Atest_str)
        self.assertTrue(test_str1 != test_strA)
        self.assertTrue(test_str1 != Test_str)

    def test_less_than(self):
        ab = UID("ab")
        abc = UID("abc")
        bc = UID("bc")
        Ab = UID("Ab")

        self.assertTrue(ab < abc)
        self.assertTrue(ab < bc)
        self.assertTrue(Ab < ab)

        self.assertTrue(not abc < ab)
        self.assertTrue(not bc < ab)
        self.assertTrue(not ab < Ab)
        self.assertTrue(not ab < ab)
