// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/metadata_traits.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace kwiver::vital;

PYBIND11_MODULE( metadata_traits, m )
{
  py::class_< metadata_tag_traits,
    std::shared_ptr< metadata_tag_traits > >( m, "MetadataTagTraits" )
    .def( "tag",         &metadata_tag_traits::tag )
    .def( "name",        &metadata_tag_traits::name )
    .def( "enum_name",   &metadata_tag_traits::enum_name )
    .def(
      "type",        []( metadata_tag_traits const& self ) -> string_t {
        if( self.type() == typeid( string_t ) )
        {
          return "string";
        }
        return self.type_name();
      } )
    .def( "description", &metadata_tag_traits::description );

  m.def( "tag_traits_by_tag",       &tag_traits_by_tag, py::arg( "tag" ) );
  m.def( "tag_traits_by_name",      &tag_traits_by_name, py::arg( "name" ) );
  m.def(
    "tag_traits_by_enum_name", &tag_traits_by_enum_name,
    py::arg( "enum_name" ) );
}
