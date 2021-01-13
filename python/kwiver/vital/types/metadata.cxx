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

#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>
#include <vital/types/metadata_tags.h>
#include <vital/types/geo_point.h>
#include <vital/types/geo_polygon.h>
#include <vital/util/demangle.h>

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace kwiver::vital;

void adder(metadata &self, py::object data, vital_metadata_tag t )
{

  #define TAG_CASE( TAG, NAME, TYPE, ... ) case VITAL_META_##TAG: \
  { \
  auto vital_meta_data_ ## TAG = data.cast< typename vital_meta_trait< VITAL_META_ ## TAG >::type >(); \
  self.add<VITAL_META_ ## TAG>(vital_meta_data_ ## TAG); \
  break; \
  }

  switch (t)
  {

    KWIVER_VITAL_METADATA_TAGS( TAG_CASE )

  default:
    // default to unknown tag type
    {
      auto vital_meta_data_ = data.cast< typename vital_meta_trait< VITAL_META_UNKNOWN >::type >();
      self.add<VITAL_META_UNKNOWN>(vital_meta_data_ );
      break;
    }
  } // end switch

#undef TAG_CASE
}

#define REGISTER_TYPED_METADATA( TAG, NAME, T, ... ) \
  py::class_< typed_metadata< VITAL_META_ ## TAG, T >, \
              std::shared_ptr< typed_metadata< VITAL_META_ ## TAG, T > >, \
              metadata_item >( m, "TypedMetadata_" #TAG ) \
  .def( py::init( [] ( std::string name, T const& data ) \
  { \
    return typed_metadata< VITAL_META_ ## TAG, T >( name, any( data ) ); \
  })) \
  .def_property_readonly( "data", [] ( typed_metadata< VITAL_META_ ## TAG, T > const& self ) \
  { \
    any dat = self.data(); \
    return any_cast< T >( dat ); \
  }) \
  .def( "as_string", [] ( typed_metadata< VITAL_META_ ## TAG, T > const& self ) \
  { \
    std::string ret = self.as_string(); \
\
    if ( typeid( T ) == typeid( bool ) ) \
    { \
      return ( ret == std::string( "1" ) ? std::string( "True" ) : std::string( "False" ) ); \
    } \
    return ret; \
  }) \
  ;

PYBIND11_MODULE( metadata, m )
{
  py::class_< metadata_item, std::shared_ptr< metadata_item > >( m, "MetadataItem" )
  .def( "is_valid",    &metadata_item::is_valid )
  .def( "__nonzero__", &metadata_item::is_valid )
  .def( "__bool__",    &metadata_item::is_valid )
  .def_property_readonly( "name", &metadata_item::name )
  .def_property_readonly( "tag",  &metadata_item::tag )
  .def_property_readonly( "type", [] ( metadata_item const& self )
  {
    // The demangled name for strings is long and complicated
    // So we'll check that case here.
    if ( self.has_string() )
    {
      return std::string( "string" );
    }
    return demangle( self.type().name() );
  })
  // NOTE: data() is put into the derived class.
  // Otherwise, we'd have to create a separate data() function for each type.

  .def( "as_double",  &metadata_item::as_double )
  .def( "has_double", &metadata_item::has_double )
  .def( "as_uint64",  &metadata_item::as_uint64 )
  .def( "has_uint64", &metadata_item::has_uint64 )
  .def( "as_string",  &metadata_item::as_string )
  .def( "has_string", &metadata_item::has_string )

  // No print_value() since it is almost the same as as_string,
  // except it accepts a stream as argument, which can be pre-configured
  // with a certain precision. Python users obviously won't be able to use this,
  // so we'll just bind as_string.
  ;

  // Now bind all of metadata_item subclasses
  // First the "unknown" type
  py::class_< unknown_metadata_item,
              std::shared_ptr< unknown_metadata_item >,
              metadata_item >( m, "UnknownMetadataItem" )
  .def( py::init<>() )
  .def( "is_valid",    &unknown_metadata_item::is_valid )
  .def( "__nonzero__", &unknown_metadata_item::is_valid )
  .def( "__bool__",    &unknown_metadata_item::is_valid )
  .def_property_readonly( "tag",  &unknown_metadata_item::tag )
  .def_property_readonly( "data", [] ( unknown_metadata_item const& self )
  {
    any dat = self.data();
    return any_cast< int >( dat ); //data is int 0
  })
  .def( "as_string",  &unknown_metadata_item::as_string )
  .def( "as_double",  &unknown_metadata_item::as_double )
  .def( "as_uint64",  &unknown_metadata_item::as_uint64 )
  ;

  // Now the typed subclasses (around 100 of them)
  KWIVER_VITAL_METADATA_TAGS( REGISTER_TYPED_METADATA )
  // Now bind the actual metadata class
  py::class_< metadata, std::shared_ptr< metadata > >( m, "Metadata" )
  .def( py::init<>() )
  // TODO: resolve rvalue references in members https://github.com/pybind/pybind11/issues/1694
  .def( "add_copy", (void (metadata::*)(std::shared_ptr<metadata_item const>const &)) &metadata::add_copy)
  // usage: .add(identifier, tag)
  .def( "add",           &adder )
  .def( "erase",         &metadata::erase )
  .def( "has",           &metadata::has )
  .def( "find",          &metadata::find, py::return_value_policy::reference_internal )
  .def( "size",          &metadata::size )
  .def( "empty",         &metadata::empty )
  .def_static( "format_string", &metadata::format_string )
  .def_property( "timestamp",   &metadata::timestamp, &metadata::set_timestamp )
  ;

  m.def( "test_equal_content", &test_equal_content )
  ;
}
#undef REGISTER_TYPED_METADATA
