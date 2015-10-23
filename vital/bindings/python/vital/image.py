"""
ckwg +31
Copyright 2015 by Kitware, Inc.
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

Interface to VITAL image class.

"""
# -*- coding: utf-8 -*-
__author__ = 'purg'

import ctypes

from vital.util import VitalObject
import PIL


class Image (VitalObject):
    """
    vital::image interface class
    """

    @classmethod
    def from_image(cls, other_image):
        """
        Construct from another Image instance, sharing the same data memory

        :param other_image: Other image to copy from
        :type other_image: Image

        :return: New Image sharing the same data memory as the source image
        :rtype: Image

        """
        img_new = cls.VITAL_LIB.vital_image_new_from_image
        img_new.argtypes = [cls.C_TYPE_PTR]
        img_new.restype = cls.C_TYPE_PTR
        return Image.from_c_pointer(img_new(other_image._inst_ptr))


    @classmethod
    def from_pil( cls, pil_image):
        """
        Construct Image from supplied PIL image object
        """

        (img_width, img_height) = pil_image.size
        mode = pil_image.mode

        if mode == "RGB":
            img_depth = 3
            img_w_step = 1
            img_h_step = img_width;
            img_d_step = img_width * img_height
        elif mode == "BGR":
            img_depth = 3
            img_w_step = 1
            img_h_step = img_width;
            img_d_step = img_width * img_height * -1
        elif mode == "L":  # 8 bit greyscale
            img_depth = 1
            img_w_step = 1
            img_h_step = img_width;
            img_d_step = 0
        else:
            raise RuntimeError("Unsupported image format.")

        # need first pixel address
        img_first_pixel = 0 # TBD

        img_new = cls.VITAL_LIB.vital_image_new_from_data
        img_new.argtypes = [ctypes.c_void_p,
                            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        img_new.restype = cls.C_TYPE_PTR

        return Image.from_c_pointer(img_new(ctypes.byref( pil_image.tostring() ),
                                            img_width, img_height, img_depth,
                                            img_w_step, img_h_step, img_d_step ) )

    def __init__(self, width=None, height=None, depth=1, interleave=False):
        """ Construct an empty image of no, or defined, dimensions.

        If width or height are None, we construct and return an empty image of
        uninitialized size.

        """
        super(Image, self).__init__()

        if width is None or height is None:
            img_new = self.VITAL_LIB.vital_image_new
            img_new.restype = self.C_TYPE_PTR
            self._inst_ptr = img_new()
        else:
            img_new = self.VITAL_LIB.vital_image_new_with_dim
            img_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t,
                                ctypes.c_size_t, ctypes.c_bool]
            img_new.restype = self.C_TYPE_PTR
            self._inst_ptr = img_new(width, height, depth, interleave)

        if not bool(self._inst_ptr):
            raise RuntimeError("Failed to construct Image instance")

    def _destroy(self):
        img_destroy = self.VITAL_LIB.vital_image_destroy
        img_destroy.argtypes = [self.C_TYPE_PTR]
        img_destroy(self._inst_ptr)

    def size(self):
        """ Get the number of bytes allocated in the given image

        :return: The number of bytes allocated in the given image
        :rtype: int

        """
        img_size = self.VITAL_LIB.vital_image_size
        img_size.argtypes = [self.C_TYPE_PTR]
        img_size.restype = ctypes.c_size_t
        return img_size(self._inst_ptr)

    def get_pil_image(self):
        """ Get image in python friendly format
        Assumptions are that the image has byte pixels.

        :return: numpy array containing image
        :rtype: pil image
        """
        img_width = self.VITAL_LIB.vital_image_width
        img_height = self.VITAL_LIB.vital_image_height
        img_depth = self.VITAL_LIB.vital_image_depth
        img_first_byte = self.VITAL_LIB.vital_image_first_byte
        # img_w_step = self.VITAL_LIB.vital_image_w_step
        img_h_step = self.VITAL_LIB.vital_image_h_step
        img_d_step = self.VITAL_LIB.vital_image_d_step

        """ It may be possible to use frombuffer() to share image memory
        if image pixels are in a buffer objecct, which is not the case now.
        """

        if img_depth == 3:
            if img_d_step == (img_width * img_height):
                mode = "RGB"
            elif img_d_step < 0:
                mode = "BGR"
            else:
                raise RuntimeError("Unsupported image format.")


            pil_image = PIL.Image.fromstring(mode, (img_width, img_height), img_first_byte,
                                         "raw", mode, img_h_step, 1 )
        elif img_depth == 1:
            pil_image = PIL.Image.fromstring("L", (img_width, img_height), img_first_byte,
                                         "raw", "L", img_h_step, 1 )
        else:
            raise RuntimeError("Unsupported image depth.")

        return pil_image
