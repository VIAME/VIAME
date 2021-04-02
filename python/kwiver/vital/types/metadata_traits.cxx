// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/metadata_traits.h>
#include <vital/util/demangle.h>

#include <pybind11/pybind11.h>
#include <type_traits>
#include <memory>
#include <string>

namespace py = pybind11;
using namespace kwiver::vital;

#define REGISTER_VTIAL_META_TRAITS( TAG, NAME, T, ...) \
  py::class_< vital_meta_trait<VITAL_META_ ## TAG >, \
              std::shared_ptr<  vital_meta_trait< VITAL_META_ ## TAG > >\
            >(m, "VitalMetaTrait_" # TAG) \
  .def( "name", &vital_meta_trait<VITAL_META_ ## TAG>::name) \
  .def( "description", &vital_meta_trait<VITAL_META_ ## TAG>::description) \
  .def( "tag_type", ([]() \
  { \
      const std::type_info * tag_type = &vital_meta_trait< VITAL_META_ ## TAG >::tag_type(); \
      if ( *tag_type == typeid( std::string )) \
      { \
        return std::string( "string" ); \
      } \
      return demangle(tag_type->name()); \
  })) \
  .def( "is_integral", &vital_meta_trait<VITAL_META_ ## TAG>::is_integral) \
  .def( "is_floating_point", &vital_meta_trait<VITAL_META_ ## TAG>::is_floating_point) \
  .def( "tag", &vital_meta_trait<VITAL_META_ ## TAG>::tag) \
  ;

PYBIND11_MODULE( metadata_traits, m )
{

  py::class_< vital_meta_trait_base,
              std::shared_ptr< vital_meta_trait_base > >( m, "VitalMetaTraitBase" )
  .def_property_readonly( "name", &vital_meta_trait_base::name )
  .def_property_readonly( "description", &vital_meta_trait_base::description )
  .def_property_readonly( "tag", &vital_meta_trait_base::tag )
  .def_property_readonly( "tag_type", [] ( vital_meta_trait_base const& self )
  {
    if ( self.tag_type() == typeid( std::string ))
    {
      return std::string( "string" );
    }
    return demangle( self.tag_type().name() );
  })
  .def( "is_integral", &vital_meta_trait_base::is_integral )
  .def( "is_floating_point", &vital_meta_trait_base::is_floating_point )
  // Note that we are binding create_metadata_item in the subclasses
  ;

  KWIVER_VITAL_METADATA_TAGS( REGISTER_VTIAL_META_TRAITS )

  py::class_< metadata_traits,
              std::shared_ptr<metadata_traits> >( m, "MetadataTraits" )
  .def( py::init<>() )
  .def( "find", &metadata_traits::find, py::return_value_policy::reference_internal)
  .def( "tag_to_symbol",      &metadata_traits::tag_to_symbol )
  .def( "tag_to_name",        &metadata_traits::tag_to_name )
  .def( "tag_to_description", &metadata_traits::tag_to_description )
  ;
}

#undef REGISTER_VITAL_META_TRAITS
