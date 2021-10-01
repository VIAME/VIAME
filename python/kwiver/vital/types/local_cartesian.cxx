// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/local_cartesian.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py=pybind11;
namespace kv=kwiver::vital;

// Wrap convert_to_cartesian
// Getting pybind to allow pass by reference with eigen data types is difficult
// Instead, call convert_to_cartesian on the C++ side and return the result,
// instead of modifying by reference.
kv::vector_3d
local_cartesian_convert_to_cartesian( kv::local_cartesian const& self, kv::geo_point const& loc )
{
  kv::vector_3d cart_out;
  self.convert_to_cartesian( loc, cart_out );
  return cart_out;
}

PYBIND11_MODULE(local_cartesian, m)
{
  py::class_< kwiver::vital::local_cartesian, std::shared_ptr< kwiver::vital::local_cartesian > >( m, "LocalCartesian" )
  .def( py::init< kv::geo_point const&, double >(), py::arg("origin"), py::arg("orientation") = 0 )
  .def( "set_origin", &kv::local_cartesian::set_origin, py::arg("origin"), py::arg("orientation") = 0 )
  .def( "get_origin", &kv::local_cartesian::get_origin )
  .def( "get_orientation", &kv::local_cartesian::get_orientation )
  // Convert_from_cartesian does not need to be wrapped.
  // Non eigen types can be passed by reference from Python
  .def( "convert_from_cartesian", &kv::local_cartesian::convert_from_cartesian )
  .def( "convert_to_cartesian", &local_cartesian_convert_to_cartesian )
  ;
}
