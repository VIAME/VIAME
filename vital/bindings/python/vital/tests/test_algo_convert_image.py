"""
ckwg +31
Copyright 2015-2016 by Kitware, Inc.
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

Tests for vital::algo::convert_image and general algorithm tests

"""
# -*- coding: utf-8 -*-
__author__ = 'paul.tunison@kitware.com'

import ctypes

from vital import (
    apm,
    ConfigBlock,
)
from vital.algo import ConvertImage
from vital.exceptions.base import VitalNullPointerException

import nose.tools as nt


def mem_address(inst_ptr):
    return int(bool(inst_ptr)) and ctypes.addressof(inst_ptr.contents)


class TestVitalAlgoConvertImage (object):
    """
    Doubles as the tests for generic algorithm and algorithm_def methods as this
    is an algorithm with a core implementation.
    """

    @classmethod
    def setup_class(cls):
        apm.register_plugins()

    def test_from_c_ptr_null(self):
        ci = ConvertImage(
            name='ci',
            from_cptr=ConvertImage.C_TYPE_PTR(),
        )
        nt.assert_false(ci.c_pointer)

    def test_from_c_ptr_no_name(self):
        nt.assert_raises(
            TypeError,
            ConvertImage,
            from_cptr=ConvertImage.C_TYPE_PTR(),
        )

        nt.assert_raises(
            ValueError,
            ConvertImage,
            None
        )

    def test_create_invalid(self):
        nt.assert_raises(
            VitalNullPointerException,
            ConvertImage.create,
            'ci', 'notAnImpl'
        )

    def test_new(self):
        ci = ConvertImage('ci')
        nt.assert_false(ci)
        nt.assert_false(ci.c_pointer)

    def test_typename(self):
        ci = ConvertImage('ci')
        nt.assert_equal(ci.type_name(), "convert_image")

    def test_name(self):
        name = 'algo_name'
        ci = ConvertImage(name)
        nt.assert_equal(ci.name, name)

    def test_set_name(self):
        algo_name = 'ci'
        other_name = "other_name"

        ci = ConvertImage(algo_name)
        nt.assert_equal(ci.name, algo_name)
        ci.set_name(other_name)
        nt.assert_not_equal(ci.name, algo_name)
        nt.assert_equal(ci.name, other_name)

    def test_impl_name(self):
        ci_empty = ConvertImage('ci')
        nt.assert_is_none(ci_empty.impl_name())

    def test_clone_empty(self):
        ci_empty = ConvertImage('ci')
        ci_empty2 = ci_empty.clone()
        nt.assert_false(ci_empty)
        nt.assert_false(ci_empty2)

    def test_clone(self):
        # inst_ptr will be null for both
        ci1 = ConvertImage('ci')
        ci2 = ci1.clone()
        nt.assert_false(ci1)
        nt.assert_false(ci2)
        nt.assert_not_equal(ci1.c_pointer, ci2.c_pointer)
        # They should both be null
        nt.assert_equal(mem_address(ci1.c_pointer), mem_address(ci2.c_pointer))

    def test_get_conf(self):
        ci = ConvertImage('ci')
        c = ci.get_config()
        nt.assert_list_equal(c.available_keys(), ['ci:type'])
        nt.assert_true(c.has_value('ci:type'))
        nt.assert_equal(c.get_value('ci:type'), '')

    def test_set_conf(self):
        ci = ConvertImage('ci')
        nt.assert_false(ci)
        nt.assert_is_none(ci.impl_name())

    def test_check_conf(self):
        ci = ConvertImage('ci')
        c = ConfigBlock()
        nt.assert_false(ci.check_config(c))

        c.set_value('ci:type', '')
        nt.assert_false(ci.check_config(c))

        c.set_value('ci:type', 'not_an_impl')
        nt.assert_false(ci.check_config(c))
