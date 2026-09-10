// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/point.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <sstream>

namespace py = pybind11;
namespace kv = kwiver::vital;

// Easy way to automate bindings of templated classes.
// For more information, see below link
// https://stackoverflow.com/questions/47487888/pybind11-template-class-of-many-types
template < size_t N, typename T >
void
declare_point( py::module& m, std::string const& typestr )
{
  using Class = kv::point< N, T >;
  using vector_type = Eigen::Matrix< T, N, 1 >;
  using covariance_type = kv::covariance_< N, float >;

  py::module::import( "kwiver.vital.types.covariance" );

  const std::string pyclass_name = std::string( "Point" ) + typestr;

  py::class_< Class,
    std::shared_ptr< Class > > p( m, pyclass_name.c_str() );
  p.def( py::init<>() );
  p.def(
    py::init< vector_type const&, covariance_type const& >(),
    py::arg( "value" ),
    py::arg_v(
      "covariance", covariance_type{},
      ( "Covar" + typestr + "()" ).c_str() ) );
  p.def(
    "__str__", []( Class const& self ){
      std::stringstream s;
      s << self;
      return ( s.str() );
    } );

  p.def_property( "value",      &Class::value,      &Class::set_value );
  p.def_property( "covariance", &Class::covariance, &Class::set_covariance );
  p.def_property_readonly(
    "type_name", [ typestr ]( Class const& /* self */ ){
      return typestr;
    } );
}

// Specialized binding for 2D points with x, y constructor
template < typename T >
void
declare_point2( py::module& m, std::string const& typestr )
{
  using Class = kv::point< 2, T >;
  using vector_type = Eigen::Matrix< T, 2, 1 >;
  using covariance_type = kv::covariance_< 2, float >;

  py::module::import( "kwiver.vital.types.covariance" );

  const std::string pyclass_name = std::string( "Point" ) + typestr;

  py::class_< Class,
    std::shared_ptr< Class > > p( m, pyclass_name.c_str() );
  p.def( py::init<>() );
  // Constructor taking x, y coordinates directly
  p.def( py::init< T, T >(), py::arg( "x" ), py::arg( "y" ) );
  p.def(
    py::init< vector_type const&, covariance_type const& >(),
    py::arg( "value" ),
    py::arg_v(
      "covariance", covariance_type{},
      ( "Covar" + typestr + "()" ).c_str() ) );
  p.def(
    "__str__", []( Class const& self ){
      std::stringstream s;
      s << self;
      return ( s.str() );
    } );

  p.def_property( "value",      &Class::value,      &Class::set_value );
  p.def_property( "covariance", &Class::covariance, &Class::set_covariance );
  p.def_property_readonly(
    "type_name", [ typestr ]( Class const& /* self */ ){
      return typestr;
    } );
}

PYBIND11_MODULE( point, m )
{
  // Use specialized 2D bindings with x,y constructor
  declare_point2< int >( m, "2i" );
  declare_point2< double >( m, "2d" );
  declare_point2< float >( m, "2f" );
  // 3D and 4D use generic binding
  declare_point< 3, double >( m, "3d" );
  declare_point< 3, float >( m, "3f" );
  declare_point< 4, double >( m, "4d" );
  declare_point< 4, float >( m, "4f" );
}
