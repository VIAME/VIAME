// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/uid.h>

#include <pybind11/pybind11.h>

namespace py=pybind11;

PYBIND11_MODULE(uid, m)
{
  py::class_<kwiver::vital::uid, std::shared_ptr<kwiver::vital::uid>>(m, "UID")
  .def(py::init<>())
  .def(py::init<const std::string&>())
  .def(py::init<const char*, size_t>())
  .def("is_valid", &kwiver::vital::uid::is_valid)
  .def("value", &kwiver::vital::uid::value)
  .def("size", &kwiver::vital::uid::size)
  .def("__len__", &kwiver::vital::uid::size)
  .def("__eq__", &kwiver::vital::uid::operator==)
  .def("__ne__", &kwiver::vital::uid::operator!=)
  .def("__lt__", &kwiver::vital::uid::operator<)
  ;
}
