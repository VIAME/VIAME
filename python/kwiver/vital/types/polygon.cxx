// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/polygon.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <memory>

namespace py=pybind11;
namespace kv=kwiver::vital;

PYBIND11_MODULE(polygon, m)
{
  py::class_< kv::polygon, std::shared_ptr< kv::polygon > >(m, "Polygon")
  .def(py::init<>())
  .def(py::init< const std::vector< kv::polygon::point_t >&>())
  .def("push_back", (void (kv::polygon::*) (double, double)) (&kv::polygon::push_back))
  .def("push_back", (void (kv::polygon::*) (const kv::polygon::point_t&)) (&kv::polygon::push_back))
  .def("num_vertices", &kv::polygon::num_vertices)
  .def("get_vertices", &kv::polygon::get_vertices)
  .def("contains", (bool (kv::polygon::*) (double, double)) (&kv::polygon::contains))
  .def("contains", (bool (kv::polygon::*) (const kv::polygon::point_t&)) (&kv::polygon::contains))
  .def("at", &kv::polygon::at)
  ;
}
