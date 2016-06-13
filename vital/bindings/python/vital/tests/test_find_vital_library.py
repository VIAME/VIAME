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

Test utilities for finding the Vital C interface library.

"""
import ctypes
import re
import sys
import unittest

import nose.tools

import vital.util.find_vital_library as fvl


__author__ = 'paul.tunison@kitware.com'


if sys.platform == 'linux2':

    class TestVitalLibraryFinding (unittest.TestCase):

        def setUp(self):
            # Clear caches
            fvl.__LIBRARY_PATH_CACHE__ = None
            fvl.__LIBRARY_CACHE__ = None

            self.orig_lib_name = fvl.__LIBRARY_NAME__
            self.orig_lib_re = fvl.__LIBRARY_NAME_RE__

            # c library should exist since on a linux system
            self.lib_re = re.compile(
                fvl.__LIBRARY_NAME_RE_BASE__ % "c"
            )

        def tearDown(self):
            fvl.__LIBRARY_PATH_CACHE__ = None
            fvl.__LIBRARY_CACHE__ = None

            fvl.__LIBRARY_NAME__ = self.orig_lib_name
            fvl.__LIBRARY_NAME_RE__ = self.orig_lib_re

        def test_search_up_dir(self):
            p = fvl._search_up_directory("/usr/local/bin", self.lib_re)
            # If not None, we found a path for the c library
            nose.tools.assert_is_not_none(p)
            nose.tools.assert_true('libc.so' in p)

            # What if from current directory
            p = fvl._search_up_directory(".", self.lib_re)
            # If not None, we found a path for the c library
            nose.tools.assert_is_not_none(p)
            nose.tools.assert_true('libc.so' in p)

            # Library name that should not exist
            r = re.compile(
                fvl.__LIBRARY_NAME_RE_BASE__
                % "not_actually_a_library_probably.6"
            )
            p = fvl._search_up_directory("/usr/local", r)
            nose.tools.assert_is_none(p)

        def test_find_path_caching(self):
            nose.tools.assert_is_none(fvl.__LIBRARY_PATH_CACHE__)
            p = fvl.find_vital_library_path()
            nose.tools.assert_is_not_none(p)
            nose.tools.assert_is_not_none(fvl.__LIBRARY_PATH_CACHE__)

        def test_find_path_no_caching(self):
            nose.tools.assert_is_none(fvl.__LIBRARY_PATH_CACHE__)
            p = fvl.find_vital_library_path(use_cache=False)
            nose.tools.assert_is_not_none(p)
            nose.tools.assert_is_none(fvl.__LIBRARY_PATH_CACHE__)

        def test_find_lib_caching(self):
            nose.tools.assert_is_none(fvl.__LIBRARY_PATH_CACHE__)
            l = fvl.find_vital_library()
            nose.tools.assert_is_instance(l, ctypes.CDLL)
            nose.tools.assert_is_not_none(fvl.__LIBRARY_CACHE__)

        def test_find_lib_no_caching(self):
            nose.tools.assert_is_none(fvl.__LIBRARY_PATH_CACHE__)
            l = fvl.find_vital_library(use_cache=False)
            nose.tools.assert_is_instance(l, ctypes.CDLL)
            nose.tools.assert_is_none(fvl.__LIBRARY_CACHE__)
