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

Test Python interface to vital::image

"""
# -*- coding: utf-8 -*-
import nose.tools

from vital.types import Image
import ctypes


__author__ = 'paul.tunison@kitware.com'


class TestVitalImage (object):

    def test_new(self):
        img = Image()

    def test_new_sized(self):
        img = Image(720, 480)

    def test_new_type(self):
        # allocated a uint32_t image
        img = Image(720, 480, 3, True, Image.PIXEL_UNSIGNED, 4)

    def test_copy_from(self):
        img = Image(720, 480, 3, True, Image.PIXEL_FLOAT, 4)
        img2 = Image().copy_from(img)
        nose.tools.assert_equal(img.equal_content(img2), True)

    def test_size(self):
        img = Image()
        nose.tools.assert_equal(img.size(), 0)

        img = Image(720, 480)
        nose.tools.assert_equal(img.size(), 720*480)

    def test_getitem_uint8(self):
        img = Image(720, 480)
        nose.tools.assert_equal(img.pixel_type_name(), "uint8")
        val1 = img[0,0]
        val2 = img[0,0,0]
        nose.tools.assert_equal(val1, val2)

    def test_getitem_int32(self):
        img = Image(720, 480, 3, True, Image.PIXEL_SIGNED, 4)
        nose.tools.assert_equal(img.pixel_type_name(), "int32")
        val1 = img[0,0]
        val2 = img[0,0,0]
        nose.tools.assert_equal(val1, val2)

    def test_getitem_float(self):
        img = Image(720, 480, 3, True, Image.PIXEL_FLOAT, 4)
        nose.tools.assert_equal(img.pixel_type_name(), "float")
        val1 = img[0,0]
        val2 = img[0,0,0]
        nose.tools.assert_equal(val1, val2)

    def test_getitem_double(self):
        img = Image(720, 480, 3, True, Image.PIXEL_FLOAT, 8)
        nose.tools.assert_equal(img.pixel_type_name(), "double")
        val1 = img[0,0]
        val2 = img[0,0,0]
        nose.tools.assert_equal(val1, val2)

    def test_getitem_bool(self):
        img = Image(720, 480, 1, True, Image.PIXEL_BOOL, 1)
        nose.tools.assert_equal(img.pixel_type_name(), "bool")
        val1 = img[0,0]
        val2 = img[0,0,0]
        nose.tools.assert_equal(val1, val2)


    def test_pil_L(self):
        # test uint8 image
        img = Image(720, 480)
        pil_img = img.get_pil_image()
        img2 = Image.from_pil(pil_img)
        nose.tools.assert_equal(img.equal_content(img2), True)

    def test_pil_F(self):
        # test float image
        img = Image(720, 480, 1, True, Image.PIXEL_FLOAT, 4)
        pil_img = img.get_pil_image()
        img2 = Image.from_pil(pil_img)
        nose.tools.assert_equal(img.equal_content(img2), True)

    def test_pil_RGB(self):
        # test RGB image
        img = Image(720, 480, 3, True)
        pil_img = img.get_pil_image()
        img2 = Image.from_pil(pil_img)
        nose.tools.assert_equal(img.equal_content(img2), True)

    def test_pil_RGBA(self):
        # test RGBA image
        img = Image(720, 480, 4, True)
        pil_img = img.get_pil_image()
        img2 = Image.from_pil(pil_img)
        nose.tools.assert_equal(img.equal_content(img2), True)

    def test_pil_I(self):
        # test int image
        img = Image(720, 480, 1, True, Image.PIXEL_SIGNED, 4)
        pil_img = img.get_pil_image()
        img2 = Image.from_pil(pil_img)
        nose.tools.assert_equal(img.equal_content(img2), True)

    def test_pil_1(self):
        # test bool image
        img = Image(720, 480, 1, True, Image.PIXEL_BOOL, 1)
        pil_img = img.get_pil_image()
        img2 = Image.from_pil(pil_img)
        nose.tools.assert_equal(img.equal_content(img2), True)
