/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#include <vital/types/image.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef kwiver::vital::image image_t;
typedef kwiver::vital::image_pixel_traits pixel_traits;

pixel_traits::pixel_type
pixel_type(std::shared_ptr<image_t> &self)
{
  auto traits = self->pixel_traits();
  return traits.type;
}

std::string
pixel_type_name(std::shared_ptr<image_t> &self)
{
  auto traits = self->pixel_traits();
  pixel_traits::pixel_type type = traits.type;
  size_t bytes = traits.num_bytes;
  std::string name = "";
  if (type == pixel_traits::pixel_type::BOOL && bytes == 1)
  {
    return "bool";
  }
  else if (type == pixel_traits::pixel_type::UNSIGNED)
  {
    name = "uint";
  }
  else if (type == pixel_traits::pixel_type::SIGNED)
  {
    name = "int";
  }
  else if (type == pixel_traits::pixel_type::FLOAT && bytes == 4)
  {
    return "float";
  }
  else if (type == pixel_traits::pixel_type::FLOAT && bytes == 8)
  {
    return "double";
  }
  return name + std::to_string(bytes*8);
}

size_t
pixel_num_bytes(std::shared_ptr<image_t> &self)
{
  auto traits = self->pixel_traits();
  return traits.num_bytes;
}

py::object
get_pixel2(std::shared_ptr<image_t> &img, unsigned i, unsigned j)
{
  std::string type = pixel_type_name(img);

  // We need to make sure we cast it from the correct type of pixel
  #define QUOTE(X) #X
  #define GET_PIXEL(TYPE, NAME) \
  if(type == QUOTE(NAME)) \
  return py::cast<TYPE>(img->at<TYPE>(i,j));

  GET_PIXEL(uint8_t, uint8)
  GET_PIXEL(int8_t, int8)
  GET_PIXEL(uint16_t, uint16)
  GET_PIXEL(int16_t, int16)
  GET_PIXEL(uint32_t, uint32)
  GET_PIXEL(int32_t, int32)
  GET_PIXEL(uint64_t, uint64)
  GET_PIXEL(int64_t, int64)
  GET_PIXEL(float, float)
  GET_PIXEL(double, double)
  GET_PIXEL(bool, bool)


  #undef GET_PIXEL
  #undef QUOTE

  return py::none();

}

py::object
get_pixel3(std::shared_ptr<image_t> &img, unsigned i, unsigned j, unsigned k)
{
  std::string type = pixel_type_name(img);

  // We need to make sure we cast it from the correct type of pixel
  #define QUOTE(X) #X
  #define GET_PIXEL(TYPE, NAME) \
  if(type == QUOTE(NAME)) \
  return py::cast<TYPE>(img->at<TYPE>(i,j,k));

  GET_PIXEL(uint8_t, uint8)
  GET_PIXEL(int8_t, int8)
  GET_PIXEL(uint16_t, uint16)
  GET_PIXEL(int16_t, int16)
  GET_PIXEL(uint32_t, uint32)
  GET_PIXEL(int32_t, int32)
  GET_PIXEL(uint64_t, uint64)
  GET_PIXEL(int64_t, int64)
  GET_PIXEL(float, float)
  GET_PIXEL(double, double)
  GET_PIXEL(bool, bool)


  #undef GET_PIXEL
  #undef QUOTE

  return py::none();

}

// __getitem__ has 2 or 3 dimensions, each calling a different function
// so the index has to be passed in as a vector
py::object
get_pixel(std::shared_ptr<image_t> &img, std::vector<unsigned> idx)
{
  if(idx.size() == 2)
  {
    return get_pixel2(img, idx[0], idx[1]);
  }
  else if(idx.size() == 3)
  {
    return get_pixel3(img, idx[0], idx[1], idx[2]);
  }
  return py::none();
}

void*
first_pixel(std::shared_ptr<image_t> &img)
{
  return img->first_pixel();
}

image_t
new_image(size_t width, size_t height, size_t depth, bool interleave,
          pixel_traits::pixel_type &type, size_t bytes)
{
  pixel_traits traits(type, bytes);
  return image_t(width, height, depth, interleave, traits);
}

image_t
new_image_from_data( char* first_pixel,
                     size_t width, size_t height, size_t depth,
                     int32_t w_step, int32_t h_step, int32_t d_step,
                     pixel_traits::pixel_type pixel_type,
                     size_t bytes)
{
  pixel_traits traits(pixel_type, bytes);
  image_t img = image_t(first_pixel, width, height, depth,
                        w_step, h_step, d_step, traits);
  image_t new_img = image_t(); // copy so we can use fresh memory not used elsewhere
  new_img.copy_from(img);
  return new_img;
}

PYBIND11_MODULE(image, m)
{
  py::class_<image_t, std::shared_ptr<image_t>> img(m, "Image", py::buffer_protocol());

  py::enum_<pixel_traits::pixel_type>(img, "Types")
  .value("PIXEL_UNKNOWN", pixel_traits::pixel_type::UNKNOWN)
  .value("PIXEL_BOOL", pixel_traits::pixel_type::BOOL)
  .value("PIXEL_UNSIGNED", pixel_traits::pixel_type::UNSIGNED)
  .value("PIXEL_SIGNED", pixel_traits::pixel_type::SIGNED)
  .value("PIXEL_FLOAT", pixel_traits::pixel_type::FLOAT)
  .export_values();

  img.def(py::init(&new_image),
    py::arg("width")=0, py::arg("height")=0, py::arg("depth")=1,
    py::arg("interleave")=false, py::arg("pixel_type")=pixel_traits::pixel_type::UNSIGNED,
    py::arg("bytes")=1)
  .def(py::init(&new_image_from_data),
   py::arg("first_pixel"), py::arg("width"), py::arg("height"), py::arg("depth"),
   py::arg("w_step"), py::arg("h_step"), py::arg("d_step"),
   py::arg("pixel_type"), py::arg("bytes"))
  .def("copy_from", &image_t::copy_from,
    py::arg("other"))
  .def("size", &image_t::size)
  .def("width", &image_t::width)
  .def("height", &image_t::height)
  .def("depth", &image_t::depth)
  .def("w_step", &image_t::w_step)
  .def("h_step", &image_t::h_step)
  .def("d_step", &image_t::d_step)
  .def("first_pixel_address", &first_pixel)
  .def("pixel_type", &pixel_type)
  .def("pixel_type_name", &pixel_type_name)
  .def("pixel_num_bytes", &pixel_num_bytes)
  .def("__getitem__", &get_pixel)
  .def_buffer([](image_t &img) -> py::buffer_info
       {
         return py::buffer_info(img.first_pixel(), 1, "B", img.size());
       })
  ;
}
