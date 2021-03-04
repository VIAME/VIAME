/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
 * DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION ) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


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
