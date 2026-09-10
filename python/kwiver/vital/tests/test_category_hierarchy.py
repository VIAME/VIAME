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

Tests for Python interface to vital::category_hierarchy

"""

from kwiver.vital.types import CategoryHierarchy

import os
import tempfile
import unittest


class TestVitalCategoryHierarchy(unittest.TestCase):
    def setUp(self):
        """
        First create the following hierarchy using lists

                        class0
                        /   \
                       /     \
                  class1_0  class1_1
                    /
                   /
                 class2_0

        where class0   has id 0,
              class1_0 has id 1,
              class1_1 has id 2, and
              class2_0 has id 3
        """
        self.class_names = ["class0", "class1_0", "class1_1", "class2_0"]
        self.parent_names = ["", "class0", "class0", "class1_0"]
        self.ids = [0, 1, 2, 3]

        # Now write to a file to create a similar hierarchy
        # Unfortunately, in order for this to work on Windows, we can't
        # utilize tempfile's automatic cleanup, as the C++ process won't be
        # able to read the file if it's still open in Python
        # So create a file and manually delete in tearDown()
        self.fp = tempfile.NamedTemporaryFile(mode="w+", delete=False)

        # This hierarchy is the same as the one constructed using lists,
        # Except class2_0 also has class1_1 as a parent. Each class
        # also has 2 synonyms of the form:
        # {classname}_syn{syn_num}, where syn_num is 0 or 1
        self.fp.writelines(
            [
                "class0 class0_syn0 class0_syn1",
                "\nclass1_0 :parent=class0 class1_0_syn0 class1_0_syn1",
                "\nclass1_1 class1_1_syn0 class1_1_syn1 :parent=class0",
                "\nclass2_0 class2_0_syn0 :parent=class1_0 :parent=class1_1 class2_0_syn1",
                "\n#class5",
            ]
        )
        self.fp.flush()

        # Close so C++ can read
        self.fp.close()

    # Manually delete the file
    def tearDown(self):
        os.remove(self.fp.name)
        self.assertFalse(os.path.exists(self.fp.name))

    def test_default_constructor(self):
        CategoryHierarchy()

    def test_construct_from_file(self):
        CategoryHierarchy(self.fp.name)

    def test_constructor_from_file_no_exist(self):
        expected_err_msg = "Unable to open nonexistant_file.txt"

        with self.assertRaisesRegex(RuntimeError, expected_err_msg):
            CategoryHierarchy("nonexistant_file.txt")

    def test_construct_from_lists(self):
        # Should be able to call with just class_names
        CategoryHierarchy(self.class_names)

        # class_names and parent_names
        CategoryHierarchy(self.class_names, self.parent_names)

        # class_names and ids
        CategoryHierarchy(self.class_names, ids=self.ids)

        # and all 3
        CategoryHierarchy(self.class_names, self.parent_names, self.ids)

    def _create_hierarchies(self):
        empty = CategoryHierarchy()
        from_file = CategoryHierarchy(self.fp.name)
        from_lists = CategoryHierarchy(self.class_names, self.parent_names, self.ids)
        return (empty, from_file, from_lists)

    def test_constructor_throws_exceptions(self):
        # Passing class_names and parent_names of different sizes
        with self.assertRaisesRegex(ValueError, "Parameter vector sizes differ."):
            CategoryHierarchy(self.class_names, self.parent_names[:-1])

        with self.assertRaisesRegex(ValueError, "Parameter vector sizes differ."):
            CategoryHierarchy(self.class_names[:-1], self.parent_names)

        with self.assertRaisesRegex(ValueError, "Parameter vector sizes differ."):
            CategoryHierarchy([], self.parent_names)

        # Passing class_names and ids of different sizes
        with self.assertRaisesRegex(ValueError, "Parameter vector sizes differ."):
            CategoryHierarchy(self.class_names, ids=self.ids[:-1])

        with self.assertRaisesRegex(ValueError, "Parameter vector sizes differ."):
            CategoryHierarchy(self.class_names[:-1], ids=self.ids)

        with self.assertRaisesRegex(ValueError, "Parameter vector sizes differ."):
            CategoryHierarchy([], ids=self.ids)

        # Passing empty class_names also throws exception
        with self.assertRaisesRegex(ValueError, "Parameter vector are empty."):
            CategoryHierarchy([])

    def test_initial_classes(self):
        empty, from_file, from_lists = self._create_hierarchies()
        # First check that each hierarchy does/does not
        # have the expected class names
        for name, id_ in zip(self.class_names, self.ids):
            # empty
            self.assertFalse(
                empty.has_class_name(name),
                "Empty hierarchy had classname {}".format(name),
            )

            # from_file
            self.assertTrue(
                from_file.has_class_name(name),
                "heirarchy constructed from file does not have {}".format(name),
            )
            self.assertEqual(from_file.get_class_id(name), id_)
            self.assertEqual(from_file.get_class_name(name), name)

            # from_lists
            self.assertTrue(
                from_lists.has_class_name(name),
                "heirarchy constructed from lists does not have {}".format(name),
            )
            self.assertEqual(from_lists.get_class_id(name), id_)
            self.assertEqual(from_lists.get_class_name(name), name)

        # Tests for empty
        self.assertEqual(empty.all_class_names(), [])
        self.assertEqual(empty.size(), 0)

        # Tests for from_file
        self.assertEqual(from_file.all_class_names(), self.class_names)
        # Each class has 2 synonyms, so size is 3 * # classes
        self.assertEqual(from_file.size(), 3 * len(self.class_names))
        # Make sure class5, which was commented out, is not present
        self.assertFalse(from_file.has_class_name("class5"))

        # Tests for from_lists
        self.assertEqual(from_lists.all_class_names(), self.class_names)
        self.assertEqual(from_lists.size(), len(self.class_names))

    # Only hierarchies constructed from files can be constructed with synonyms
    def test_initial_synonyms(self):
        ch = CategoryHierarchy(self.fp.name)

        for cname in ch.all_class_names():
            syn0_name = cname + "_syn0"
            syn1_name = cname + "_syn1"
            self.assertTrue(ch.has_class_name(syn0_name))
            self.assertTrue(ch.has_class_name(syn1_name))
            self.assertEqual(ch.get_class_name(syn0_name), cname)
            self.assertEqual(ch.get_class_name(syn1_name), cname)

    def test_initial_relationships(self):
        empty, from_file, from_lists = self._create_hierarchies()

        # Tests for empty
        self.assertEqual(empty.child_class_names(), [])

        # Tests for from_file
        self.assertEqual(from_file.child_class_names(), ["class2_0"])
        self.assertEqual(from_file.get_class_parents("class0"), [])
        self.assertEqual(
            from_file.get_class_parents("class2_0"), ["class1_0", "class1_1"]
        )
        self.assertEqual(from_file.get_class_parents("class1_0"), ["class0"])
        self.assertEqual(from_file.get_class_parents("class1_1"), ["class0"])

        # Tests for from_lists
        self.assertEqual(from_lists.child_class_names(), ["class1_1", "class2_0"])
        self.assertEqual(from_lists.get_class_parents("class0"), [])
        self.assertEqual(from_lists.get_class_parents("class2_0"), ["class1_0"])
        self.assertEqual(from_lists.get_class_parents("class1_0"), ["class0"])
        self.assertEqual(from_lists.get_class_parents("class1_1"), ["class0"])

    def test_add_class(self):
        ch = CategoryHierarchy()

        # Check default for parent_name and id params
        ch.add_class("class0")
        self.assertEqual(ch.get_class_id("class0"), -1)

        # Now for parent_name
        ch.add_class("class1", id=0)
        self.assertEqual(ch.get_class_id("class1"), 0)

        # Now for id
        ch.add_class("class2", parent_name="class1")
        self.assertEqual(ch.get_class_id("class2"), -1)

        # Check has_class_name returns correct result
        self.assertTrue(ch.has_class_name("class0"))
        self.assertTrue(ch.has_class_name("class1"))
        self.assertTrue(ch.has_class_name("class2"))

        # Check class list
        self.assertEqual(ch.all_class_names(), ["class1", "class0", "class2"])
        self.assertEqual(ch.size(), 3)

        # Check relationships are correct
        # TODO: Should this only be class2 and class0? Current implementation
        # of add_class only adds class1 to class2's parents. Class2 isn't added
        # to Class1's list of children, which makes it a child class.
        self.assertEqual(ch.child_class_names(), ["class1", "class0", "class2"])

        self.assertEqual(ch.get_class_parents("class0"), [])
        self.assertEqual(ch.get_class_parents("class1"), [])
        self.assertEqual(ch.get_class_parents("class2"), ["class1"])

    def test_add_class_already_exists(self):
        ch = CategoryHierarchy(self.class_names, self.parent_names, self.ids)

        with self.assertRaisesRegex(RuntimeError, "Category already exists"):
            ch.add_class(self.class_names[0])

        ch.add_class("new_class")

        with self.assertRaisesRegex(RuntimeError, "Category already exists"):
            ch.add_class("new_class")

    def test_add_relationship(self):
        ch = CategoryHierarchy()
        ch.add_class("class0")
        ch.add_class("class1_0")
        ch.add_class("class1_1")
        ch.add_class("class2_0")

        # Same as the file
        ch.add_relationship("class1_0", "class0")
        ch.add_relationship("class1_1", "class0")
        ch.add_relationship("class2_0", "class1_0")
        ch.add_relationship("class2_0", "class1_1")

        self.assertEqual(ch.child_class_names(), ["class2_0"])
        self.assertEqual(ch.get_class_parents("class2_0"), ["class1_0", "class1_1"])
        self.assertEqual(ch.get_class_parents("class1_0"), ["class0"])
        self.assertEqual(ch.get_class_parents("class1_1"), ["class0"])

    def test_add_synonym(self):
        ch = CategoryHierarchy(self.class_names, self.parent_names, self.ids)

        ch.add_synonym("class2_0", "class2_0_syn0")
        ch.add_synonym("class2_0", "class2_0_syn1")
        ch.add_synonym("class1_0", "class1_0_syn0")
        ch.add_synonym("class1_0", "class1_0_syn1")

        # First check the old classes exist
        self.assertEqual(ch.all_class_names(), self.class_names)

        # Check the size
        self.assertEqual(ch.size(), 8)

        # Now check synonyms exist
        self.assertTrue(ch.has_class_name("class2_0_syn0"))
        self.assertTrue(ch.has_class_name("class2_0_syn1"))
        self.assertTrue(ch.has_class_name("class1_0_syn0"))
        self.assertTrue(ch.has_class_name("class1_0_syn1"))

        # Check the name of the actual category
        self.assertEqual(ch.get_class_name("class2_0_syn0"), "class2_0")
        self.assertEqual(ch.get_class_name("class2_0_syn1"), "class2_0")
        self.assertEqual(ch.get_class_name("class1_0_syn0"), "class1_0")
        self.assertEqual(ch.get_class_name("class1_0_syn1"), "class1_0")

        # Now check that the relationships are still intact
        self.assertEqual(ch.get_class_parents("class2_0_syn0"), ["class1_0"])
        self.assertEqual(ch.get_class_parents("class2_0_syn1"), ["class1_0"])
        self.assertEqual(ch.get_class_parents("class1_0_syn0"), ["class0"])
        self.assertEqual(ch.get_class_parents("class1_0_syn1"), ["class0"])

    def test_add_synonym_already_exists(self):
        ch = CategoryHierarchy()
        ch.add_class("class0")
        ch.add_synonym("class0", "class0_syn0")
        ch.add_synonym("class0", "class0_syn1")

        expected_err_msg = "Synonym name already exists in hierarchy"

        with self.assertRaisesRegex(RuntimeError, expected_err_msg):
            ch.add_synonym("class0", "class0_syn0")

        with self.assertRaisesRegex(RuntimeError, expected_err_msg):
            ch.add_synonym("class0", "class0_syn1")

    def test_load_from_file(self):
        ch = CategoryHierarchy()
        ch.add_class("class-1")
        ch.add_synonym("class-1", "class-1_syn0")
        ch.add_synonym("class-1", "class-1_syn1")

        ch.load_from_file(self.fp.name)
        self.assertEqual(ch.all_class_names(), self.class_names + ["class-1"])

        # Check synonyms
        for cname in self.class_names + ["class-1"]:
            self.assertTrue(ch.has_class_name(cname + "_syn0"))
            self.assertTrue(ch.has_class_name(cname + "_syn1"))
            self.assertEqual(ch.get_class_name(cname + "_syn0"), cname)
            self.assertEqual(ch.get_class_name(cname + "_syn1"), cname)

        # Now check that the relationships are still intact
        self.assertEqual(ch.child_class_names(), ["class2_0", "class-1"])
        self.assertEqual(ch.get_class_parents("class0"), [])
        self.assertEqual(ch.get_class_parents("class2_0"), ["class1_0", "class1_1"])
        self.assertEqual(ch.get_class_parents("class1_0"), ["class0"])
        self.assertEqual(ch.get_class_parents("class1_1"), ["class0"])

    def test_load_from_file_not_exist(self):
        ch = CategoryHierarchy()
        expected_err_msg = "Unable to open nonexistant_file.txt"

        with self.assertRaisesRegex(RuntimeError, expected_err_msg):
            ch.load_from_file("nonexistant_file.txt")

    # Some functions throw exceptions if the category
    # can't be found. Those will be tested here
    def test_category_not_exist(self):
        chs = list(self._create_hierarchies())
        expected_err_msg = "Class node absent_class does not exist"

        for ch in chs:
            with self.assertRaisesRegex(RuntimeError, expected_err_msg):
                ch.add_class("new_class1", "absent_class")

            with self.assertRaisesRegex(RuntimeError, expected_err_msg):
                ch.get_class_name("absent_class")

            with self.assertRaisesRegex(RuntimeError, expected_err_msg):
                ch.get_class_id("absent_class")

            with self.assertRaisesRegex(RuntimeError, expected_err_msg):
                ch.get_class_parents("absent_class")

            with self.assertRaisesRegex(RuntimeError, expected_err_msg):
                ch.add_relationship("absent_class", "another_absent_class")

            ch.add_class("new_class2")
            with self.assertRaisesRegex(RuntimeError, expected_err_msg):
                ch.add_relationship("new_class2", "absent_class")

            with self.assertRaisesRegex(RuntimeError, expected_err_msg):
                ch.add_synonym("absent_class", "synonym")

    # Extra test for the sort function used
    # in a few member functions. all_class_names()
    # essentially returns the result
    def test_sort(self):
        ch = CategoryHierarchy()

        # Adding them in this way forces
        # every type of comparison to be made
        ch.add_class("a", id=1)
        ch.add_class("c")
        ch.add_class("b")
        ch.add_class("d", id=0)

        # names with ids are first sorted (in alphabetical order), followed by
        # names without ids in alphabetical order
        self.assertEqual(ch.all_class_names(), ["d", "a", "b", "c"])
        self.assertEqual(ch.child_class_names(), ["d", "a", "b", "c"])
