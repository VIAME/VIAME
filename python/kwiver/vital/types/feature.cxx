// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/feature.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <sstream>

namespace py=pybind11;
namespace kv=kwiver::vital;
namespace kwiver {
namespace vital  {
namespace python {

// Easy way to automate bindings of templated classes.
// For more information, see below link
// https://stackoverflow.com/questions/47487888/pybind11-template-class-of-many-types
template< typename T >
void declare_feature( py::module &m, std::string const& typestr )
{
  using Class = kv::feature_< T >;
  const std::string pyclass_name = std::string( "Feature" ) + typestr;

  py::class_< Class,
              std::shared_ptr< Class >,
              kv::feature >( m, pyclass_name.c_str() )
  .def( py::init<>() )
  .def( py::init< kv::feature const& >() )
  .def( py::init< Eigen::Matrix< T, 2, 1 > const&, T, T, T, kv::rgb_color const& >(),
    py::arg( "loc" ),
    py::arg( "mag" ) = 0.0,
    py::arg( "scale" ) = 1.0,
    py::arg( "angle" ) = 0.0,
    py::arg( "rgb_color" ) = kv::rgb_color() )
  .def( "clone", &Class::clone )
  .def( "__str__", [] ( const Class& self )
  {
    std::stringstream s;
    s << self;
    return s.str();
  })

  .def_property( "location",   &Class::get_loc,       &Class::set_loc )
  .def_property( "magnitude",  &Class::get_magnitude, &Class::set_magnitude )
  .def_property( "scale",      &Class::get_scale,     &Class::set_scale )
  .def_property( "angle",      &Class::get_angle,     &Class::set_angle )
  .def_property( "covariance", &Class::get_covar,     &Class::set_covar )
  .def_property( "color",      &Class::get_color,     &Class::set_color )
  .def_property_readonly( "type_name", [] ( const Class& self )
  {
    return self.data_type().name()[0];
  })

  ;
}
}
}
}

using namespace kwiver::vital::python;
PYBIND11_MODULE(feature, m)
{
  py::class_< kv::feature, std::shared_ptr< kv::feature > >( m, "Feature" )
  .def( "__eq__", [] ( const kv::feature& self, const kv::feature& other )
    {
      return self == other;
    })
  .def( "equal_except_for_angle", [] ( const kv::feature& self, const kv::feature& other )
    {
      return self.equal_except_for_angle( other );
    })
  .def( "__ne__", [] ( const kv::feature& self, const kv::feature& other )
    {
      return self != other;
    })
  ;
  declare_feature< float  >( m, "F" );
  declare_feature< double >( m, "D" );
}
