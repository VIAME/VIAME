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

Interface to VITAL image class.

"""
# -*- coding: utf-8 -*-
__author__ = 'paul.tunison@kitware.com'

import ctypes
from vital.util import VitalObject


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
        return Image(from_cptr=img_new(other_image._inst_ptr))

    @classmethod
    def from_pil(cls, pil_image):
        """
        Construct Image from supplied PIL image object
        """

        (img_width, img_height) = pil_image.size
        mode = pil_image.mode

        if mode == "RGB":
            img_depth = 3
            img_w_step = 3
            img_h_step = img_width * 3
            img_d_step = -1
        elif mode == "L":  # 8 bit greyscale
            img_depth = 1
            img_w_step = 1
            img_h_step = img_width
            img_d_step = 0
        else:
            raise RuntimeError("Unsupported image format.")

        img_new = cls.VITAL_LIB.vital_image_new_from_data
        img_new.argtypes = [ctypes.c_char_p,
                            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        img_new.restype = cls.C_TYPE_PTR

        return Image(from_cptr=img_new(pil_image.tostring(),
                                       img_width, img_height, img_depth,
                                       img_w_step, img_h_step, img_d_step))

    # TODO: Need to add class-method from_numpy( cls, numpy_arry )

    def __init__(self, width=None, height=None, depth=1, interleave=False,
                 from_cptr=None):
        """
        Construct an empty image of no, or defined, dimensions.

        If width or height are None, we construct and return an empty image of
        uninitialized size.

        """
        super(Image, self).__init__(from_cptr, width, height, depth, interleave)

    def _new(self, width, height, depth, interleave):
        if width is None or height is None:
            img_new = self.VITAL_LIB.vital_image_new
            img_new.restype = self.C_TYPE_PTR
            return img_new()
        else:
            img_new = self.VITAL_LIB.vital_image_new_with_dim
            img_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t,
                                ctypes.c_size_t, ctypes.c_bool]
            img_new.restype = self.C_TYPE_PTR
            return img_new(width, height, depth, interleave)

    def _destroy(self):
        img_destroy = self.VITAL_LIB.vital_image_destroy
        img_destroy.argtypes = [self.C_TYPE_PTR]
        img_destroy(self._inst_ptr)

    #
    # ----------------------------------------------------------
    # Accessors

    def size(self):
        """ Get the number of bytes allocated in the given image

        :return: The number of bytes allocated in the given image
        :rtype: int

        """
        img_size = self.VITAL_LIB.vital_image_size
        img_size.argtypes = [self.C_TYPE_PTR]
        img_size.restype = ctypes.c_size_t
        return img_size(self._inst_ptr)

    def width(self):
        """
        Get the image width in pixels
        """
        return self.VITAL_LIB.vital_image_width(self)

    def height(self):
        """
        Get image height in pixels
        """
        return self.VITAL_LIB.vital_image_height(self)

    def depth(self):
        """
        Get image depth in planes
        """
        return self.VITAL_LIB.vital_image_depth(self)

    def first_pixel_address(self):
        """
        Get the address of the first pixel in the image
        """
        first_pixel = self.VITAL_LIB.vital_image_first_pixel
        first_pixel.restype = ctypes.c_void_p
        return first_pixel(self)

    def w_step(self):
        """
        Get the step value to go to next column
        """
        return self.VITAL_LIB.vital_image_w_step(self)

    def h_step(self):
        """
        Get the step value to go to next row
        """
        return self.VITAL_LIB.vital_image_h_step(self)

    def d_step(self):
        """
        Get the step value to go to next plane
        """
        return self.VITAL_LIB.vital_image_d_step(self)



    # ------------------------------------------------------------------
    #+# Make a utility method not a member
    # or a derived class :: pil_image_converter
    # def get_pil_image( Image ): returns PIL image
    # need to check type of input
    def get_pil_image(self):
        """ Get image in python friendly format
        Assumptions are that the image has byte pixels.

        :return: numpy array containing image
        :rtype: pil image
        """
        import PIL.Image as PIM
        img_first_byte = self.first_pixel_address()

        # get buffer from image
        pixels = ctypes.pythonapi.PyBuffer_FromReadWriteMemory
        pixels.argtypes = [ ctypes.c_void_p, ctypes.c_int ]
        pixels.restype = ctypes.py_object
        img_pixels = pixels( img_first_byte, self.size() )

        # determine image format from strides
        if self.depth() == 3:
            if self.d_step() == 1:
                mode = "BGR"
            elif self.d_step() < 0:
                mode = "RGB"
            else:
                raise RuntimeError("Unsupported image format.")

            pil_image = PIM.frombytes("RGB", (self.width(), self.height()), img_pixels,
                                      "raw", mode, self.h_step(), 1 )
        elif self.depth() == 1:
            pil_image = PIM.frombytes("L", (self.width(), self.height()), img_pixels,
                                      "raw", "L", self.h_step(), 1 )
        else:
            raise RuntimeError("Unsupported image depth.")

        return pil_image
