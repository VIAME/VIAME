"""
ckwg +31
Copyright 2017 by Kitware, Inc.
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

Helper functions for dealing with PIL

"""

from kwiver.vital.types import Image
import six

def _pil_image_to_bytes(p_img):
    """
    Get the component bytes from the given PIL Image.
    In recent version of PIL, the tobytes function is the correct thing to
    call, but some older versions of PIL do not have this function.
    :param p_img: PIL Image to get the bytes from.
    :type p_img: PIL.Image.Image
    :returns: Byte string.
    :rtype: bytes
    """
    if hasattr(p_img, 'tobytes'):
        return p_img.tobytes()
    else:
        # Older version of the function.
        return p_img.tostring()


def _pil_image_from_bytes(mode, size, data, decoder_name='raw', *args):
    """
    Creates a copy of an image memory from pixel data in a buffer.
    In recent versionf of PIL, the frombytes function is the correct thing to
    call, but older version fo PIL only have a fromstring, which is equivalent
    in function.
    :param mode: The image mode. See: :ref:`concept-modes`.
    :param size: The image size.
    :param data: A byte buffer containing raw data for the given mode.
    :param decoder_name: What decoder to use.
    :param args: Additional parameters for the given decoder.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    import PIL.Image
    if hasattr(PIL.Image, 'frombytes'):
        return PIL.Image.frombytes(mode, size, data, decoder_name, *args)
    else:
        return PIL.Image.fromstring(mode, size, data, decoder_name, *args)

def from_pil(pil_image):
    """
    Construct Image from supplied PIL image object.
    :param pil_image: PIL image object
    :type pil_image: PIL.Image.Image
    :raises RuntimeError: If the PIL Image provided is not in a recognized
       mode.
    :returns: New Image instance using the given image's pixels.
    :rtype: Image
    """

    (img_width, img_height) = pil_image.size
    mode = pil_image.mode
    # TODO(paul.tunison): Extract this logic out into a utility function.
    if mode == "1":  # boolean
        img_depth = 1
        img_w_step = 1
        img_h_step = img_width
        img_d_step = 0
        img_pix_num_bytes = 1
        img_pix_type = Image.PIXEL_BOOL
    elif mode == "L":  # 8-bit greyscale
        img_depth = 1
        img_w_step = 1
        img_h_step = img_width
        img_d_step = 0
        img_pix_num_bytes = 1
        img_pix_type = Image.PIXEL_UNSIGNED
    elif mode == "RGB": # 8-bit RGB
        img_depth = 3
        img_w_step = 3
        img_h_step = img_width * 3
        img_d_step = 1
        img_pix_num_bytes = 1
        img_pix_type = Image.PIXEL_UNSIGNED
    elif mode == "RGBA":  # 8-bit RGB with alpha
        img_depth = 4
        img_w_step = 4
        img_h_step = img_width * 4
        img_d_step = 1
        img_pix_num_bytes = 1
        img_pix_type = Image.PIXEL_UNSIGNED
    elif mode == "I":  # 32-bit signed int greyscale
        img_depth = 1
        img_w_step = 1
        img_h_step = img_width
        img_d_step = 0
        img_pix_num_bytes = 4
        img_pix_type = Image.PIXEL_SIGNED
    elif mode == "F":  # 32-bit float greyscale
        img_depth = 1
        img_w_step = 1
        img_h_step = img_width
        img_d_step = 0
        img_pix_num_bytes = 4
        img_pix_type = Image.PIXEL_FLOAT
    else:
        raise RuntimeError("Unsupported image format.")

    img_data = _pil_image_to_bytes(pil_image)
    vital_img = Image(img_data,
                      img_width, img_height, img_depth,
                      img_w_step, img_h_step, img_d_step,
                      img_pix_type, img_pix_num_bytes)
    return vital_img

def get_pil_image(img):
    """ Get image in python friendly format
    Assumptions are that the image has byte pixels.
    :return: array containing image
    :rtype: pil image
    """
    def pil_mode_from_image(img):
        """
        Determine image format from pixel properties
        May return None if our current encoding does not map to a PIL image
        mode.
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

    mode = pil_mode_from_image(img)

    if not mode:
        # make a copy of this image using contiguous memory with interleaved channels
        new_img = Image(img.width(), img.height(), img.depth(),
                        True, img.pixel_type(), img.pixel_num_bytes())
        new_img.copy_from(img)
        img = new_img
        mode = pil_mode_from_image(img)

    if not mode:
        raise RuntimeError("Unsupported image format.")

    # get buffer from image
    if six.PY2:
        img_pixels = buffer(bytearray(img))
    else:
        img_pixels = memoryview(bytearray(img)).tobytes()

    pil_img = _pil_image_from_bytes(mode, (img.width(), img.height()),
                                    img_pixels, "raw", mode,
                                    img.h_step() * img.pixel_num_bytes(), 1)
    return pil_img
