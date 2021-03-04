/*ckwg +29
 * Copyright 2017-2020 by Kitware, Inc.
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
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
