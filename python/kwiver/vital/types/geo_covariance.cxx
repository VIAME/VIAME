// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/geo_covariance.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py=pybind11;
namespace kv=kwiver::vital;

PYBIND11_MODULE( geo_covariance, m )
{
  // This is needed to subclass geo_point
  py::module::import( "kwiver.vital.types.geo_point" );

  py::class_< kwiver::vital::geo_covariance,
              std::shared_ptr< kwiver::vital::geo_covariance >,
              kv::geo_point >( m, "GeoCovariance" )
  .def( py::init<>() )
  .def( py::init< kv::geo_point::geo_2d_point_t const&, int >() )
  .def( py::init< kv::geo_point::geo_3d_point_t const&, int >() )
  .def_property( "covariance", &kv::geo_covariance::covariance, &kv::geo_covariance::set_covariance )
  .def( "__str__", [] ( const kv::geo_covariance& self )
  {
    std::stringstream res;
    res << self;
    return res.str();
  })
  ;
}
