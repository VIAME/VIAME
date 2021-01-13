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
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
