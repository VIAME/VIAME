// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/geo_covariance.h>

#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include "python_fold.h"

namespace py = pybind11;
namespace kv = viame;

PYBIND11_MODULE( geo_covariance, m )
{
  // This is needed to subclass geo_point
  VIAME_PYTHON_REQUIRE( "viame.types.geo_point" );

  py::class_< viame::geo_covariance,
    std::shared_ptr< viame::geo_covariance >,
    kv::geo_point >( m, "GeoCovariance" )
    .def( py::init<>() )
    .def( py::init< kv::geo_point::geo_2d_point_t const&, int >() )
    .def( py::init< kv::geo_point::geo_3d_point_t const&, int >() )
    .def_property(
      "covariance", &kv::geo_covariance::covariance,
      &kv::geo_covariance::set_covariance )
    .def(
      "__str__", []( const kv::geo_covariance& self ){
        std::stringstream res;
        res << self;
        return res.str();
      } )
  ;
}
