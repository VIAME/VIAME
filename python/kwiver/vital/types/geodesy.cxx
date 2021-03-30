// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/geodesy.h>

#include <python/kwiver/vital/util/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;

PYBIND11_MODULE( geodesy, m )
{
  // Define a submodule for the SRID namespace,
  // which contains identifier codes for different spatial reference systems
  auto msri = m.def_submodule( "SRID" );
  msri.attr( "lat_lon_NAD83" ) =       kv::SRID::lat_lon_NAD83;
  msri.attr( "lat_lon_WGS84" ) =       kv::SRID::lat_lon_WGS84;
  msri.attr( "UPS_WGS84_north" ) =     kv::SRID::UPS_WGS84_north;
  msri.attr( "UPS_WGS84_south" ) =     kv::SRID::UPS_WGS84_south;
  msri.attr( "UTM_WGS84_north" ) =     kv::SRID::UTM_WGS84_north;
  msri.attr( "UTM_WGS84_south" ) =     kv::SRID::UTM_WGS84_south;
  msri.attr( "UTM_NAD83_northeast" ) = kv::SRID::UTM_NAD83_northeast;
  msri.attr( "UTM_NAD83_northwest" ) = kv::SRID::UTM_NAD83_northwest;
  ;

  // Free functions for conversion.
  // The geo_conversion class itself isn't bound because of its protected virtual destructor,
  // which pybind complains about. We'll just bind the important free functions that wrap
  // the functor
  m.def( "geo_crs_description", &kv::geo_crs_description );
  m.def( "geo_conv", (( kv::vector_2d (*) ( kv::vector_2d const&, int, int )) &kv::geo_conv ));
  m.def( "geo_conv", (( kv::vector_3d (*) ( kv::vector_3d const&, int, int )) &kv::geo_conv ));

  // utm_ups_zone_t
  py::class_< kv::utm_ups_zone_t, std::shared_ptr<kv::utm_ups_zone_t >>( m, "UTMUPSZone" )
  .def( py::init(( kv::utm_ups_zone_t (*) ( double, double )) &kv::utm_ups_zone ))
  .def( py::init(( kv::utm_ups_zone_t (*) ( kv::vector_2d const& lon_lat )) &kv::utm_ups_zone ))
  .def( py::init(( kv::utm_ups_zone_t (*) ( kv::vector_3d const& lon_lat_alt )) &kv::utm_ups_zone ))
  .def_readwrite( "number", &kv::utm_ups_zone_t::number )
  .def_readwrite( "north", &kv::utm_ups_zone_t::north )
  ;
}
