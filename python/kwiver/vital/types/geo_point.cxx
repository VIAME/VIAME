// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/geo_point.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <sstream>
#include <memory>

namespace kv=kwiver::vital;
namespace py=pybind11;

PYBIND11_MODULE(geo_point, m)
{
  py::class_<kv::geo_point, std::shared_ptr<kv::geo_point>>(m, "GeoPoint")
  .def( py::init<>() )
  .def( py::init<kv::geo_point::geo_2d_point_t const&, int>() )
  .def( py::init<kv::geo_point::geo_3d_point_t const&, int>() )
  .def( "crs", &kv::geo_point::crs )
  .def( "location",
    ( kv::geo_point::geo_3d_point_t ( kv::geo_point::*) () const )
    ( &kv::geo_point::location ))
  .def( "location",
    ( kv::geo_point::geo_3d_point_t ( kv::geo_point::*) ( int ) const )
    ( &kv::geo_point::location ))
  .def( "set_location",
    ( void ( kv::geo_point::* ) ( kv::geo_point::geo_2d_point_t const&, int ))
    ( &kv::geo_point::set_location ))
  .def( "set_location",
    ( void ( kv::geo_point::* ) ( kv::geo_point::geo_3d_point_t const&, int ))
    ( &kv::geo_point::set_location ))
  .def( "is_empty", &kv::geo_point::is_empty )
  .def( "__str__", [] ( const kv::geo_point& self )
  {
    std::stringstream res;
    res << self;
    return res.str();
  })
  ;
}
