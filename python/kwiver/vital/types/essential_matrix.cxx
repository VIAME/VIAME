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

#include <vital/types/essential_matrix.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <sstream>

namespace py=pybind11;
namespace kv=kwiver::vital;

// Easy way to automate bindings of templated classes.
// For more information, see below link
// https://stackoverflow.com/questions/47487888/pybind11-template-class-of-many-types
template< typename T >
void declare_essential_matrix( py::module &m,
                               std::string const& class_typestr,
                               std::string const& dtype )
{
  using Class = kv::essential_matrix_< T >;
  const std::string pyclass_name = std::string( "EssentialMatrix" ) + class_typestr;
  typedef Eigen::Matrix< T, 3, 3 > matrix_t;
  typedef Eigen::Matrix< T, 3, 1 > vector_t;

  py::class_< Class, std::shared_ptr< Class >, kv::essential_matrix >( m, pyclass_name.c_str() )
    .def( py::init< matrix_t const& >() )
    .def( py::init< kv::rotation_<T> const&, vector_t const& >() )
    .def( py::init< Class const& >() )
    .def( py::init< kv::essential_matrix const& >() )
    .def( "clone", &Class::clone )
    .def( "matrix", &Class::matrix )
    .def( "rotation", &Class::rotation )
    .def( "twisted_rotation", &Class::twisted_rotation )
    .def( "translation", &Class::translation )
    .def( "compute_matrix", &Class::compute_matrix )
    .def( "compute_twisted_rotation", &Class::compute_twisted_rotation )
    .def( "get_rotation", &Class::get_rotation )
    .def( "get_translation", &Class::get_translation )
    .def( "__str__", [] ( Class const& self )
    {
      std::stringstream str;
      str << self;
      return str.str();
    })
    .def( "__getitem__", []( Class& self, py::tuple idx )
                      {
                        int i = idx[0].cast< int >();
                        int j = idx[1].cast< int >();
                        if( 0 <= i && i < 3 && 0 <= j && j < 3 )
                        {
                          return self.matrix()( i, j );
                        }
                        else
                        {
                          throw py::index_error( "Index out of range!" );
                        }
                      })
    .def_property_readonly("type_name", [dtype] ( Class const& self )
    {
      return dtype;
    })
  ;
}


PYBIND11_MODULE(essential_matrix, m)
{
  py::class_< kv::essential_matrix, std::shared_ptr< kv::essential_matrix > >( m, "BaseEssentialMatrix" );
  declare_essential_matrix< double >( m, "D", "d" );
  declare_essential_matrix< float  >( m, "F", "f" );
}
