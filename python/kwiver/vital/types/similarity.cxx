// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/similarity.h>

#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <memory>
namespace py = pybind11;

namespace kwiver {

namespace vital {

namespace python {

namespace kv = kwiver::vital;

template < typename T >
void
declare_similarity(
  py::module& m,
  std::string const& class_typestr,
  std::string const& dtype )
{
  using Class = kv::similarity_< T >;

  const std::string pyclass_name = std::string( "Similarity" ) + class_typestr;

  py::class_< Class, std::shared_ptr< Class > >( m, pyclass_name.c_str() )
    .def( py::init() )
    .def( py::init< kv::similarity_< float > const& >() )
    .def( py::init< kv::similarity_< double > const& >() )
    .def(
      py::init< T const&, kv::rotation_< T > const&,
        kwiver::vital::vector_< 3, T > const& >() )
    .def( py::init< kwiver::vital::matrix_< 4, 4, T > const& >() )
    .def( "matrix", &Class::matrix )
    .def( "inverse", &Class::inverse )
    .def(
      "__mul__", []( Class const& self, Class const& other ){
        return self * other;
      } )
    .def(
      "__mul__", []( Class const& self, kwiver::vital::vector_< 3, T > const& rhs ){
        return self * rhs;
      } )
    .def(
      "__eq__", []( Class const& self, Class const& other ){
        return self == other;
      } )
    .def(
      "__ne__", []( Class const& self, Class const& other ){
        return self != other;
      } )
    .def_property_readonly( "scale", &Class::scale )
    .def_property_readonly( "rotation", &Class::rotation )
    .def_property_readonly( "translation", &Class::translation )
    .def_property_readonly(
      "type_name", [ dtype ]( Class const& /* self */ ){
        return dtype;
      } )
  ;
}

} // namespace python

} // namespace vital

} // namespace kwiver

using namespace kwiver::vital::python;
PYBIND11_MODULE( similarity, m )
{
  declare_similarity< float >( m, "F", "f" );
  declare_similarity< double >( m, "D", "d" );
}
