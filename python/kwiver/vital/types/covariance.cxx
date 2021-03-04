// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/covariance.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

// Easy way to automate bindings of templated classes.
// For more information, see below link
// https://stackoverflow.com/questions/47487888/pybind11-template-class-of-many-types
template< unsigned N, typename T >
void declare_covariance( py::module &m, std::string const& typestr )
{
  using Class = kv::covariance_< N, T >;
  const std::string pyclass_name = std::string( "Covar" ) + typestr;

  py::class_< Class, std::shared_ptr< Class > >( m, pyclass_name.c_str() )
  .def( py::init<>() )
  .def( py::init< const T& >() )
  .def( py::init< const Eigen::Matrix< T, N, N >& >() )
  .def( "matrix", &Class::matrix )
  .def( "__setitem__", []( Class& self, py::tuple idx, T value )
                      {
                        // Casting values to unsigned removes compiler
                        // warning when comparing int and unsigned int.
                        // Converting right to unsigned in the .cast<>() call
                        // results in a misleading runtime error if negative
                        // values for i or j are used, thus the implicit cast
                        unsigned int i = idx[0].cast<int>();
                        unsigned int j = idx[1].cast<int>();
                        if( i >= N || j >= N )
                        {
                          throw py::index_error( "Index out of range!" );
                        }
                        self( i, j ) = value;
                      })
  .def( "__getitem__", []( Class& self, py::tuple idx )
                      {
                        unsigned int i = idx[0].cast<int>();
                        unsigned int j = idx[1].cast<int>();
                        if( i >= N || j >= N )
                        {
                          throw py::index_error( "Index out of range!" );
                        }
                        return self( i, j );
                      })
  ;
}


PYBIND11_MODULE(covariance, m)
{
  declare_covariance< 2, double >( m, "2d" );
  declare_covariance< 2, float  >( m, "2f" );
  declare_covariance< 3, double >( m, "3d" );
  declare_covariance< 3, float  >( m, "3f" );
  declare_covariance< 4, double >( m, "4d" );
  declare_covariance< 4, float  >( m, "4f" );
}
