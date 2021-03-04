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

#include <vital/types/point.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <sstream>

namespace py=pybind11;
namespace kv=kwiver::vital;

// Easy way to automate bindings of templated classes.
// For more information, see below link
// https://stackoverflow.com/questions/47487888/pybind11-template-class-of-many-types
template< unsigned N, typename T >
void declare_point( py::module &m, std::string const& typestr )
{
  using Class = kv::point< N, T >;
  using vector_type = Eigen::Matrix< T, N, 1 >;
  using covariance_type = kv::covariance_< N, float >;

  const std::string pyclass_name = std::string( "Point" ) + typestr;

  py::class_< Class,
              std::shared_ptr< Class > > p( m, pyclass_name.c_str() );
  p.def( py::init<>() );
  p.def( py::init< vector_type const&, covariance_type const& >() );
  p.def( "__str__", [] ( Class const& self )
  {
    std::stringstream s;
    s << self;
    return ( s.str() );
  });

  p.def_property( "value",      &Class::value,      &Class::set_value );
  p.def_property( "covariance", &Class::covariance, &Class::set_covariance );
  p.def_property_readonly( "type_name", [ typestr ] ( Class const& self )
  {
    return typestr;
  });
}

PYBIND11_MODULE( point, m )
{
  declare_point< 2, int >   ( m, "2i" );
  declare_point< 2, double >( m, "2d" );
  declare_point< 2, float > ( m, "2f" );
  declare_point< 3, double >( m, "3d" );
  declare_point< 3, float > ( m, "3f" );
  declare_point< 4, double >( m, "4d" );
  declare_point< 4, float > ( m, "4f" );
}
