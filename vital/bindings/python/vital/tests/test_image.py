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
import numpy as np


class TestVitalImage (object):

    def test_new(self):
        img = Image()

    def test_new_sized(self):
        img = Image(720, 480)

    def test_new_type(self):
        # allocated a uint32_t image
        img = Image(720, 480, 3, True, Image.PIXEL_UNSIGNED, 4)

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

    def test_numpy_conversion(self):
        # TODO: do pytest parametarize once we move to pytest
        dtype_names = ['bool',
                       'int8', 'int16', 'int32',
                       'uint8', 'uint16', 'uint32',
                       # 'float16',  # currently not supported
                       'float32',
                       'float64']

        def _test_numpy(dtype_name, nchannels, order='c'):
            if nchannels is None:
                shape = (5, 4)
            else:
                shape = (5, 4, nchannels)
            size = np.prod(shape)

            dtype = np.dtype(dtype_name)

            if dtype_name == 'bool':
                np_img = np.zeros(size, dtype=dtype).reshape(shape)
                np_img[0::2] = 1
            else:
                np_img = np.arange(size, dtype=dtype).reshape(shape)

            if order.startswith('c'):
                np_img = np.ascontiguousarray(np_img)
            elif order.startswith('fortran'):
                np_img = np.asfortranarray(np_img)
            else:
                raise KeyError(order)
            if order.endswith('-reverse'):
                np_img = np_img[::-1, ::-1]

            vital_img = Image(np_img)
            recast = vital_img.asarray()

            if nchannels is None:
                # asarray always returns 3 channels
                np_img = np_img[..., None]

            pixel_type_name = vital_img.pixel_type_name()

            if dtype_name == 'float16':
                want = 'float16'
            if dtype_name == 'float32':
                want = 'float'
            elif dtype_name == 'float64':
                want = 'double'
            else:
                want = dtype_name

            assert pixel_type_name == want, 'want={} but got={}'.format(
                want, pixel_type_name)

            if not np.all(np_img == recast):
                raise AssertionError(
                    'Failed dtype={}, nchannels={}, order={}'.format(
                        dtype_name, nchannels, order))

        n_pass = 0
        for order in ['c', 'fortran', 'c-reverse', 'fortran-reverse']:
            for nchannels in [None, 1, 3, 4]:
                for dtype_name in dtype_names:
                    _test_numpy(dtype_name, nchannels)
                    n_pass += 1
        # print('n_pass = {!r}'.format(n_pass))
        # vital_img = Image(np.asfortranarray(np_img))
        # assert vital_img.pixel_type_name() == 'float'

    def test_numpy_share_memory(self):
        # TODO: do pytest parametarize once we move to pytest

        np_img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        vital_img = Image(np_img)

        assert np.all(np_img == vital_img.asarray()), (
            'must be initially the same')

        np_img += 1
        assert np.all(np_img != vital_img.asarray()), (
            'we do not share memory yet')
