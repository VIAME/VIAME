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

#include <vital/types/image_container.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

typedef kwiver::vital::image_container image_cont;
typedef kwiver::vital::simple_image_container s_image_cont;

// We need to return a shared pointer--otherwise, pybind11 may lose the subtype
std::shared_ptr<s_image_cont>
new_cont(kwiver::vital::image &img)
{
  return std::shared_ptr<s_image_cont>(new s_image_cont(img));
}

// We need to do a deep copy instead of just calling get_image, so we can ref track in python
kwiver::vital::image
get_image(std::shared_ptr<image_cont> self)
{
  kwiver::vital::image img;
  img.copy_from(self->get_image());
  return img;
}

PYBIND11_MODULE(image_container, m)
{
  py::class_<image_cont, std::shared_ptr<image_cont>>(m, "BaseImageContainer")
  .def("size", &image_cont::size)
  .def("width", &image_cont::width)
  .def("height", &image_cont::height)
  .def("depth", &image_cont::depth)
  .def("image", &get_image)
  ;

  py::class_<s_image_cont, image_cont, std::shared_ptr<s_image_cont>>(m, "ImageContainer")
  .def(py::init(&new_cont),
    py::arg("image"))
  ;
}
