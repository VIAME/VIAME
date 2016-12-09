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
    PIXEL_UNKNOWN = 0
    PIXEL_UNSIGNED = 1
    PIXEL_SIGNED = 2
    PIXEL_FLOAT = 3
    PIXEL_BOOL = 4

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

        if mode == "1":  # boolean
            img_depth = 1
            img_w_step = 1
            img_h_step = img_width
            img_d_step = 0
            img_pix_num_bytes = 1
            img_pix_type = cls.PIXEL_BOOL
        elif mode == "L":  # 8-bit greyscale
            img_depth = 1
            img_w_step = 1
            img_h_step = img_width
            img_d_step = 0
            img_pix_num_bytes = 1
            img_pix_type = cls.PIXEL_UNSIGNED
        elif mode == "RGB": # 8-bit RGB
            img_depth = 3
            img_w_step = 3
            img_h_step = img_width * 3
            img_d_step = 1
            img_pix_num_bytes = 1
            img_pix_type = cls.PIXEL_UNSIGNED
        elif mode == "RGBA":  # 8-bit RGB with alpha
            img_depth = 4
            img_w_step = 4
            img_h_step = img_width * 4
            img_d_step = 1
            img_pix_num_bytes = 1
            img_pix_type = cls.PIXEL_UNSIGNED
        elif mode == "I":  # 32-bit signed int greyscale
            img_depth = 1
            img_w_step = 1
            img_h_step = img_width
            img_d_step = 0
            img_pix_num_bytes = 4
            img_pix_type = cls.PIXEL_SIGNED
        elif mode == "F":  # 32-bit float greyscale
            img_depth = 1
            img_w_step = 1
            img_h_step = img_width
            img_d_step = 0
            img_pix_num_bytes = 4
            img_pix_type = cls.PIXEL_FLOAT
        else:
            raise RuntimeError("Unsupported image format.")

        img_new = cls.VITAL_LIB.vital_image_new_from_data
        img_new.argtypes = [ctypes.c_void_p,
                            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctypes.c_int32, ctypes.c_size_t]
        img_new.restype = cls.C_TYPE_PTR

        img_data = pil_image.tostring()
        # this constructor create a wrapper around img_data which will be invalid
        # when img_data goes out of scope and is deleted
        vital_img = Image(from_cptr=img_new(img_data,
                                            img_width, img_height, img_depth,
                                            img_w_step, img_h_step, img_d_step,
                                            img_pix_type, img_pix_num_bytes))
        # return a deep copy of the image into memory managed by the image object
        return Image().copy_from(vital_img)


    # TODO: Need to add class-method from_numpy( cls, numpy_arry )

    def __init__(self, width=None, height=None, depth=1, interleave=False,
                 pix_type=PIXEL_UNSIGNED, pix_num_bytes=1,
                 from_cptr=None):
        """
        Construct an empty image of no, or defined, dimensions.

        If width or height are None, we construct and return an empty image of
        uninitialized size.

        """
        super(Image, self).__init__(from_cptr, width, height, depth,
                                    interleave, pix_type, pix_num_bytes)

    def _new(self, width, height, depth, interleave, pix_type, pix_num_bytes):
        if width is None or height is None:
            img_new = self.VITAL_LIB.vital_image_new
            img_new.restype = self.C_TYPE_PTR
            return img_new()
        else:
            img_new = self.VITAL_LIB.vital_image_new_with_dim
            img_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t,
                                ctypes.c_size_t, ctypes.c_bool,
                                ctypes.c_int32, ctypes.c_size_t]
            img_new.restype = self.C_TYPE_PTR
            return img_new(width, height, depth, interleave,
                           pix_type, pix_num_bytes)

    def _destroy(self):
        img_destroy = self.VITAL_LIB.vital_image_destroy
        img_destroy.argtypes = [self.C_TYPE_PTR]
        img_destroy(self._inst_ptr)

    def copy_from(self, other):
        """
        Deep copy the image data from another image into this one.

        If the size or type of this image does not match the other image
        then reallocate memory to match.
        """
        img_copy_from = self.VITAL_LIB.vital_image_copy_from_image
        img_copy_from.argtypes = [self.C_TYPE_PTR, self.C_TYPE_PTR]
        img_copy_from(self._inst_ptr, other._inst_ptr)
        return self

    #
    # ----------------------------------------------------------
    # Accessors

    def pixel_type_name(self):
        """ Access the name of the pixel type for common types
        """
        pt = self.pixel_type()
        nb = self.pixel_num_bytes()
        name = None
        if pt == self.PIXEL_UNSIGNED:
            name = "uint"
        elif pt == self.PIXEL_SIGNED:
            name = "int"
        elif pt == self.PIXEL_FLOAT and nb == 4:
            return "float"
        elif pt == self.PIXEL_FLOAT and nb == 8:
            return "double"
        elif pt == self.PIXEL_BOOL and nb == 1:
            return "bool"
        if name:
            return name + str(nb*8)
        return None

    def __getitem__(self, tup):
        """ Access the pixel at location i, j
        """
        typename = self.pixel_type_name()
        if not typename:
            raise RuntimeError("image does not contain known pixel type")
        if len(tup) == 2:
            i, j = tup
            img_get_pixel = getattr(self.VITAL_LIB, "vital_image_get_pixel2_"+typename)
            #img_get_pixel.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
            img_get_pixel.restype = getattr(ctypes, "c_"+typename)
            return img_get_pixel(self,i,j)
        elif len(tup) == 3:
            i, j, k = tup
            img_get_pixel = getattr(self.VITAL_LIB, "vital_image_get_pixel3_"+typename)
            img_get_pixel.restype = getattr(ctypes, "c_"+typename)
            return img_get_pixel(self, i, j, k)
        raise IndexError("expected 2 or 3 indices, got "+str(len(tup)))

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

    def pixel_num_bytes(self):
        """
        Get the number of bytes per pixel
        """
        return self.VITAL_LIB.vital_image_pixel_num_bytes(self)

    def pixel_type(self):
        """
        Return a code describing how to interpret the iamage.

        0 => Unknown
        1 => Unsigned Integer
        2 => Signed Integer
        3 => Floating Point
        4 => Boolean
        """
        return self.VITAL_LIB.vital_image_pixel_type(self)

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

    def is_contiguous(self):
        """
        Return True if the image memory is contiguous
        """
        img_is_contiguous = self.VITAL_LIB.vital_image_is_contiguous
        img_is_contiguous.restype = ctypes.c_bool
        return img_is_contiguous(self)

    def equal_content(self, other):
        """
        Return True if the image has the same content as another image
        """
        img_equal_content = self.VITAL_LIB.vital_image_equal_content
        img_equal_content.restype = ctypes.c_bool
        return img_equal_content(self, other)



    # ------------------------------------------------------------------
    # Make a utility method not a member
    # or a derived class :: pil_image_converter
    # def get_pil_image( Image ): returns PIL image
    # need to check type of input
    def get_pil_image(self):
        """ Get image in python friendly format
        Assumptions are that the image has byte pixels.

        :return: array containing image
        :rtype: pil image
        """
        import PIL.Image as PIM

        def pil_mode_from_image(img):
            """ determine image format from pixel properties
            """
            if img.pixel_type() == img.PIXEL_UNSIGNED and img.pixel_num_bytes() == 1:
                if img.depth() == 3 and img.d_step() == 1 and img.w_step() == 3:
                    return "RGB"
                elif img.depth() == 4 and img.d_step() == 1 and img.w_step() == 4:
                    return "RGBA"
                elif img.depth() == 1 and img.w_step() == 1:
                    return "L"
            elif img.depth() == 1 and img.w_step() == 1:
                if img.pixel_type() == img.PIXEL_BOOL and img.pixel_num_bytes() == 1:
                    return "1"
                elif img.pixel_type() == img.PIXEL_SIGNED and img.pixel_num_bytes() == 4:
                    return "I"
                elif img.pixel_type() == img.PIXEL_FLOAT and img.pixel_num_bytes() == 4:
                    return "F"
            return None

        img = self
        mode = pil_mode_from_image(img)

        if not mode:
            # make a copy of this image using contiguous memory with interleaved channels
            img = Image(self.width(), self.height(), self.depth(),
                        True, self.pixel_type(), self.pixel_num_bytes())
            img.copy_from(self)
            mode = pil_mode_from_image(img)

        if not mode:
            raise RuntimeError("Unsupported image format.")

        img_first_byte = img.first_pixel_address()

        size = img.size()
        if size == 0:
            size = (img.h_step() * img.height()) * img.pixel_num_bytes()
        # get buffer from image
        pixels = ctypes.pythonapi.PyBuffer_FromReadWriteMemory
        pixels.argtypes = [ ctypes.c_void_p, ctypes.c_int ]
        pixels.restype = ctypes.py_object
        img_pixels = pixels( img_first_byte, size )

        return PIM.frombytes(mode, (img.width(), img.height()), img_pixels,
                             "raw", mode, img.h_step() * img.pixel_num_bytes(), 1 )


    # ------------------------------------------------------------------
    # return image as a numpy array
    def get_numpy_array(self):
        """ Get image in python friendly format
        Assumptions are that the image has byte pixels.

        :return: numpy array containing image
        :rtype: numpy array
        """
        pil_image = get_pil_image(self)
        numpy_array = numpy.array(pil_image)
        return numpy_array


    # ------------------------------------------------------------------
    # return image as an ocv iamge
    # Convert RGB to BGR
    def get_ocv_image(self):
        """ Get image in python friendly format
        Assumptions are that the image has byte pixels.

        :return: image in openCV format
        :rtype: ocv image
        """
        numpy_array = get_numpy_array(self)
        # Convert RGB to BGR
        open_cv_image = numpy_array[:, :, ::-1].copy()
        return open_cv_image
