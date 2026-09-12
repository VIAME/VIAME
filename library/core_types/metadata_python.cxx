// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/geo_point.h>
#include <viame/core_types/geo_polygon.h>
#include <viame/core_types/metadata.h>
#include <viame/core_types/metadata_tags.h>
#include <viame/core_types/metadata_traits.h>
#include <viame/algorithm_framework/util/demangle.h>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <memory>
#include <string>

namespace py = pybind11;
using namespace kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
struct to_py_visitor
{
  template < class T >
  py::object
  operator()( T const& value ) const
  {
    return py::cast( value );
  }
};

// ----------------------------------------------------------------------------
struct from_py_visitor
{
  template < class T >
  metadata_value
  operator()() const
  {
    return object.cast< T >();
  }

  py::object const& object;
};

// ----------------------------------------------------------------------------
metadata_value
from_py( vital_metadata_tag tag, py::object data )
{
  return visit_metadata_types_return< metadata_value >(
    from_py_visitor{ data }, tag_traits_by_tag( tag ).type() );
}

// ----------------------------------------------------------------------------
py::object
to_py( metadata_value const& data )
{
  return std::visit( to_py_visitor{}, data );
}

// ----------------------------------------------------------------------------
void
adder( metadata& self, py::object data, vital_metadata_tag tag )
{
  self.add( tag, from_py( tag, data ) );
}

} // namespace <anonymous>

// ----------------------------------------------------------------------------
PYBIND11_MODULE( metadata, m )
{
  py::module::import( "kwiver.vital.types.metadata_tags" );

  py::class_< metadata_item,
    std::shared_ptr< metadata_item > >( m, "MetadataItem" )
    .def(
    py::init(
      []( vital_metadata_tag tag, py::object data ){
        return new metadata_item{ tag, from_py( tag, data ) };
      } ) )
    .def( "is_valid",    &metadata_item::is_valid )
    .def( "__nonzero__", &metadata_item::is_valid )
    .def( "__bool__",    &metadata_item::is_valid )
    .def_property_readonly( "name", &metadata_item::name )
    .def_property_readonly( "tag",  &metadata_item::tag )
    .def_property_readonly(
      "type", []( metadata_item const& self ){
        // The demangled name for strings is long and complicated
        // So we'll check that case here.
        if( self.has_string() )
        {
          return std::string( "string" );
        }
        return demangle( self.type().name() );
      } )
    .def_property_readonly(
      "data", []( metadata_item const& self ){
        return to_py( self.data() );
      } )
    .def( "as_double",  &metadata_item::as_double )
    .def( "has_double", &metadata_item::has_double )
    .def( "as_uint64",  &metadata_item::as_uint64 )
    .def( "has_uint64", &metadata_item::has_uint64 )
    .def(
      "as_string",  []( metadata_item const& self ) -> string_t {
        if( self.type() == typeid( bool ) )
        {
          return self.get< bool >() ? "True" : "False";
        }
        return self.as_string();
      } )
    .def( "has_string", &metadata_item::has_string )

  // No print_value() since it is almost the same as as_string,
  // except it accepts a stream as argument, which can be pre-configured
  // with a certain precision. Python users obviously won't be able to use this,
  // so we'll just bind as_string.
  ;

  // Now bind the actual metadata class
  py::class_< metadata, std::shared_ptr< metadata > >( m, "Metadata" )
    .def( py::init<>() )
    // TODO: resolve rvalue references in members
    // https://github.com/pybind/pybind11/issues/1694
    .def(
      "add_copy",
      ( void ( metadata::* )(
        std::shared_ptr< metadata_item const > const& ) ) & metadata::add_copy )
    // usage: .add(identifier, tag)
    .def( "add",           &adder )
    .def( "erase",         &metadata::erase )
    .def( "has",           &metadata::has )
    .def(
      "find",          &metadata::find,
      py::return_value_policy::reference_internal )
    .def( "size",          &metadata::size )
    .def( "empty",         &metadata::empty )
    .def_static( "format_string", &metadata::format_string )
    .def_property(
      "timestamp",   &metadata::timestamp,
      &metadata::set_timestamp )
    .def(
      "__iter__", []( std::shared_ptr< metadata > self){
        return py::make_iterator( self->begin(), self->end() );
      },
      py::keep_alive< 0, 1 >() /* keep object alive while iterator exists */ )
    .def_static(
      "print_metadata",
      [](py::object fileHandle, std::shared_ptr< metadata > metadata){
        std::ofstream fout;
        py::scoped_ostream_redirect stream( fout, fileHandle );
        kwiver::vital::print_metadata( fout, *metadata.get() );
      } )
  ;
}
#undef REGISTER_TYPED_METADATA
