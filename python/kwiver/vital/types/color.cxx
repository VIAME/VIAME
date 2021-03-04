// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
