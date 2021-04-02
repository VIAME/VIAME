// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/geo_polygon.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <memory>

namespace py=pybind11;
namespace kv=kwiver::vital;

PYBIND11_MODULE(geo_polygon, m)
{
  py::class_< kv::geo_polygon, std::shared_ptr< kv::geo_polygon> >(m, "GeoPolygon")
  .def(py::init<>())
  .def(py::init< kv::geo_polygon::geo_raw_polygon_t const&, int >())
  .def("polygon", (kv::geo_polygon::geo_raw_polygon_t (kv::geo_polygon::*) () const) (&kv::geo_polygon::polygon))
  .def("polygon", (kv::geo_polygon::geo_raw_polygon_t (kv::geo_polygon::*) (int) const) (&kv::geo_polygon::polygon))
  .def("crs", &kv::geo_polygon::crs)
  .def("set_polygon", &kv::geo_polygon::set_polygon)
  .def("is_empty", &kv::geo_polygon::is_empty)
  .def("__str__", [](const kv::geo_polygon& self)
  {
    std::stringstream res;
    res << self;
    return res.str();
  })
  ;
}
