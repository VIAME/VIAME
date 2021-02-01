"""
ckwg +31
Copyright 2015-2019 by Kitware, Inc.
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

Test Python interface to vital::image_container

"""
# -*- coding: utf-8 -*-

from kwiver.vital.types import (
    Image,
    ImageContainer
)

import nose.tools
import numpy as np
from kwiver.vital.tests.helpers import create_numpy_image, map_dtype_name_to_pixel_type


class TestVitalImageContainer (object):

    def test_new(self):
        image = Image()
        img_c = ImageContainer(image)

        image = Image(100, 100)
        img_c = ImageContainer(image)

    def test_size(self):
        i = Image(720, 480)
        ic = ImageContainer(i)
        nose.tools.assert_equal(ic.size(), 720 * 480)

    def test_width(self):
        i = Image(720, 480)
        ic = ImageContainer(i)
        nose.tools.assert_equal(ic.width(), 720)

    def test_height(self):
        i = Image(720, 480)
        ic = ImageContainer(i)
        nose.tools.assert_equal(ic.height(), 480)

    def test_fromarray(self):
        dtype_names = ['bool',
                       'int8', 'int16', 'int32',
                       'uint8', 'uint16', 'uint32',
                       # 'float16',  # currently not supported
                       'float32',
                       'float64']

        def _test_numpy(dtype_name, nchannels, order='c'):
            np_img = create_numpy_image(dtype_name, nchannels, order)
            img_container = ImageContainer.fromarray(np_img)
            recast = img_container.asarray()

            # asarray always returns 3 channels
            np_img = np.atleast_3d(np_img)

            vital_img = img_container.image()
            pixel_type_name = vital_img.pixel_type_name()

            pixel_type_name = vital_img.pixel_type_name()
            want = map_dtype_name_to_pixel_type(dtype_name)

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

    def test_asarray(self):
        dtype_names = ['bool',
                       'int8', 'int16', 'int32',
                       'uint8', 'uint16', 'uint32',
                       # 'float16',  # currently not supported
                       'float32',
                       'float64']

        def _test_numpy(dtype_name, nchannels, order='c'):
            np_img = create_numpy_image(dtype_name, nchannels, order)
            img_container = ImageContainer(Image(np_img))
            recast = img_container.asarray()

            # asarray always returns 3 channels
            np_img = np.atleast_3d(np_img)

            vital_img = img_container.image()
            pixel_type_name = vital_img.pixel_type_name()

            pixel_type_name = vital_img.pixel_type_name()
            want = map_dtype_name_to_pixel_type(dtype_name)

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
