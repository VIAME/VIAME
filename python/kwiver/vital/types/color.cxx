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

#include <vital/types/color.h>

#include <pybind11/pybind11.h>

namespace py=pybind11;

PYBIND11_MODULE(color, m)
{
  py::class_<kwiver::vital::rgb_color, std::shared_ptr<kwiver::vital::rgb_color> >(m, "RGBColor")
  .def(py::init<>())
  .def(py::init([](float r, float g, float b) {return kwiver::vital::rgb_color(uint8_t(r),uint8_t(g),uint8_t(b));}),
    py::arg("r")=0, py::arg("g")=0, py::arg("b")=0)
  .def_readwrite("r", &kwiver::vital::rgb_color::r)
  .def_readwrite("g", &kwiver::vital::rgb_color::g)
  .def_readwrite("b", &kwiver::vital::rgb_color::b)
  .def("__eq__", [](kwiver::vital::rgb_color self, kwiver::vital::rgb_color other)
                      {
                        return ((self.r == other.r) && (self.g == other.g) && (self.b == other.b));
                      })
  .def("__ne__", [](kwiver::vital::rgb_color self, kwiver::vital::rgb_color other)
                      {
                        return ((self.r != other.r) || (self.g != other.g) || (self.b != other.b));
                      })
  .def("__repr__", [](kwiver::vital::rgb_color self)
                      {
                        return "RGBColor{" + std::to_string(self.r) + ", " + std::to_string(self.g) + ", " + std::to_string(self.b) + "}";
                      })
  .def("__getitem__", [](kwiver::vital::rgb_color self, int idx)
                        {
                          switch (idx) {
                            case 0: return self.r;
                            case 1: return self.g;
                            case 2: return self.b;
                          }
                          throw pybind11::index_error("RGB can't have an index greater than 2");
                        })
  ;
}
