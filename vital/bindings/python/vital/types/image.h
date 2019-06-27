/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef KWIVER_VITAL_PYTHON_IMAGE_H_
#define KWIVER_VITAL_PYTHON_IMAGE_H_

#include <vital/types/image.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

typedef kwiver::vital::image image_t;
typedef kwiver::vital::image_pixel_traits pixel_traits;
namespace py = pybind11;

void image(py::module &m);

namespace kwiver {
namespace vital {
namespace python {
namespace image {
  pixel_traits::pixel_type pixel_type(std::shared_ptr<image_t> &self);
  std::string pixel_type_name(std::shared_ptr<image_t> &self);
  size_t pixel_num_bytes(std::shared_ptr<image_t> &self);
  py::object get_pixel2(std::shared_ptr<image_t> &img, unsigned i, unsigned j);
  py::object get_pixel3(std::shared_ptr<image_t> &img, unsigned i, unsigned j, unsigned k);
  py::object get_pixel(std::shared_ptr<image_t> &img, std::vector<unsigned> idx);
  void* first_pixel(std::shared_ptr<image_t> &img);
  image_t new_image(size_t width, size_t height, size_t depth, bool interleave,
            pixel_traits::pixel_type &type, size_t bytes);
  image_t new_image_from_data( char* first_pixel,
                       size_t width, size_t height, size_t depth,
                       int32_t w_step, int32_t h_step, int32_t d_step,
                       pixel_traits::pixel_type pixel_type,
                       size_t bytes);
  template <typename T>
  image_t
  new_image_from_numpy(py::array_t<T> array)
  {

    // Request a buffer descriptor from Python
    py::buffer_info info = array.request();

    // Determine if the type has numeric_limits, is integral, and / or is signed
    // Note: the following function does not handle float16. Should it?
    pixel_traits traits =  kwiver::vital::image_pixel_traits_of<T>();

    // numpy images are in height x width format by default (row major)
    size_t height = info.shape[0];
    size_t width = info.shape[1];
    size_t depth;

    size_t h_step = info.strides[0] / traits.num_bytes;
    size_t w_step = info.strides[1] / traits.num_bytes;
    size_t d_step;

    if (info.ndim == 2)
    {
        depth = 1;
        d_step = 1;
    }
    else if (info.ndim == 3)
    {
      depth = info.shape[2];
      d_step = info.strides[2]  / traits.num_bytes;
    }
    else
    {
      throw std::runtime_error("Incompatible buffer dimension!");
    }

    char* first_pixel = static_cast<char *>(info.ptr);
    image_t img = image_t(first_pixel, width, height, depth,
                          w_step, h_step, d_step, traits);
    // TODO: is is possible to share memory?
    image_t new_img = image_t(); // copy so we can use fresh memory not used elsewhere
    new_img.copy_from(img);
    return new_img;
  }
  const char* get_trait_format_descriptor(const pixel_traits& traits);
  py::buffer_info get_buffer_info(image_t &img);
  py::object asarray(image_t img);

} } } }
#endif
