// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/fundamental_matrix.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <cctype>
#include <sstream>

namespace py=pybind11;
namespace kv=kwiver::vital;

// Easy way to automate bindings of templated classes.
// For more information, see below link
// https://stackoverflow.com/questions/47487888/pybind11-template-class-of-many-types
template< typename T >
void declare_fundamental_matrix( py::module &m,
                                 std::string const& class_typestr,
                                 std::string const& dtype )
{
  using Class = kv::fundamental_matrix_< T >;
  const std::string pyclass_name = std::string( "FundamentalMatrix" ) + class_typestr;
  typedef Eigen::Matrix< T, 3, 3 > matrix_t;

  py::class_< Class, std::shared_ptr< Class >, kv::fundamental_matrix >( m, pyclass_name.c_str() )
    .def( py::init< matrix_t const& >() )
    .def( py::init< kv::fundamental_matrix const& >() )
    .def( py::init< Class const& >() )
    .def( "clone", &Class::clone )
    .def( "matrix", &Class::matrix )
    .def( "__str__", [] ( Class const& self )
    {
      std::stringstream str;
      str << self;
      return ( str.str() );
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

PYBIND11_MODULE(fundamental_matrix, m)
{
  py::class_< kv::fundamental_matrix, std::shared_ptr< kv::fundamental_matrix > >( m, "BaseFundamentalMatrix" );
  declare_fundamental_matrix< double >( m, "D", "d" );
  declare_fundamental_matrix< float  >( m, "F", "f" );
}
