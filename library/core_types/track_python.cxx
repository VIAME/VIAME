// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/attribute_set.h>
#include <viame/core_types/track.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;

namespace viame {

namespace python {

py::object
track_find_state( viame::track& self, int64_t frame_id )
{
  auto frame_itr = self.find( frame_id );
  if( frame_itr == self.end() )
  {
    throw py::index_error();
  }
  return py::cast( *frame_itr );
}

// Helper function to set an attribute from a Python object
void
track_set_attribute(
  viame::track& self, std::string const& key,
  py::object value )
{
  // Convert Python object to appropriate C++ type and store
  if( py::isinstance< py::bool_ >( value ) )
  {
    self.set_attribute( key, value.cast< bool >() );
  }
  else if( py::isinstance< py::int_ >( value ) )
  {
    self.set_attribute( key, value.cast< int64_t >() );
  }
  else if( py::isinstance< py::float_ >( value ) )
  {
    self.set_attribute( key, value.cast< double >() );
  }
  else if( py::isinstance< py::str >( value ) )
  {
    self.set_attribute( key, value.cast< std::string >() );
  }
  else
  {
    throw std::runtime_error(
      "Unsupported attribute type. Supported types: bool, int, float, str" );
  }
}

// Helper function to get an attribute as a Python object
py::object
track_get_attribute( viame::track const& self, std::string const& key )
{
  auto attrs = self.attributes();
  if( !attrs )
  {
    throw attribute_set_exception(
      "No attribute set exists for this track" );
  }
  if( !attrs->has( key ) )
  {
    throw attribute_set_exception( "Attribute not found: " + key );
  }

  auto data = attrs->data( key );

  // Try to convert to known types
  if( data.type() == typeid( bool ) )
  {
    return py::cast( viame::any_cast< bool >( data ) );
  }
  else if( data.type() == typeid( int ) )
  {
    return py::cast( viame::any_cast< int >( data ) );
  }
  else if( data.type() == typeid( int64_t ) )
  {
    return py::cast( viame::any_cast< int64_t >( data ) );
  }
  else if( data.type() == typeid( double ) )
  {
    return py::cast( viame::any_cast< double >( data ) );
  }
  else if( data.type() == typeid( std::string ) )
  {
    return py::cast( viame::any_cast< std::string >( data ) );
  }
  else
  {
    throw std::runtime_error(
      "Attribute has unsupported type for Python conversion" );
  }
}

// Helper to check if attribute exists
bool
track_has_attribute( viame::track const& self, std::string const& key )
{
  return self.has_attribute( key );
}

// Helper to get all attribute keys
std::vector< std::string >
track_attribute_keys( viame::track const& self )
{
  std::vector< std::string > keys;
  auto attrs = self.attributes();
  if( attrs )
  {
    for( auto it = attrs->begin(); it != attrs->end(); ++it )
    {
      keys.push_back( it->first );
    }
  }
  return keys;
}

} // namespace python

} // namespace viame

using namespace viame::python;
PYBIND11_MODULE( track, m )
{
  py::class_< viame::track_state,
    std::shared_ptr< viame::track_state > >( m, "TrackState" )
    .def( py::init< int64_t >(), py::arg( "frame_id" ) )
    .def( py::self == py::self, py::arg( "other" ) )
    .def_property(
      "frame_id", &viame::track_state::frame,
      &viame::track_state::set_frame )
  ;

  py::class_< viame::track,
    std::shared_ptr< viame::track > >( m, "Track" )
    .def(
    py::init(
      [](int64_t id){
        auto track = viame::track::create();
        track->set_id( id );
        return track;
      } ),
    py::arg( "id" ) = 0 )
    .def( "all_frame_ids", &viame::track::all_frame_ids )
    .def(
      "append",
      [](viame::track& self,
         std::shared_ptr< viame::track_state > track_state){
        return self.append( track_state );
      }, py::arg( "state" ) )
    .def(
      "append", [](viame::track& self, viame::track& track){
        return self.append( track );
      }, py::arg( "track" ) )
    .def( "find_state", &track_find_state, py::arg( "frame_id" ) )
    .def(
      "__iter__", [](const viame::track& self){
        return py::make_iterator( self.begin(), self.end() );
      }, py::keep_alive< 0, 1 >() )
    .def( "__len__", &viame::track::size )
    .def( "__getitem__", &track_find_state, py::arg( "frame_id" ) )
    .def_property(
      "id", &viame::track::id,
      &viame::track::set_id )
    .def_property_readonly( "size", &viame::track::size )
    .def_property_readonly( "is_empty", &viame::track::empty )
    .def_property_readonly( "first_frame", &viame::track::first_frame )
    .def_property_readonly( "last_frame", &viame::track::last_frame )
    .def(
      "set_attribute", &viame::python::track_set_attribute,
      py::arg( "key" ), py::arg( "value" ),
      R"(
      Set an attribute value for this track.

      Creates the attribute set automatically if it doesn't exist.
      This method is thread-safe.

      Args:
          key: The attribute name/key (string)
          value: The attribute value (bool, int, float, or str)

      Raises:
          RuntimeError: If value type is not supported

      Example:
          >>> track.set_attribute("species", "fish")
          >>> track.set_attribute("length_cm", 42.5)
          >>> track.set_attribute("is_verified", True)
      )" )
    .def(
      "get_attribute", &viame::python::track_get_attribute,
      py::arg( "key" ),
      R"(
      Get an attribute value from this track.

      Args:
          key: The attribute name/key (string)

      Returns:
          The attribute value (type depends on what was stored)

      Raises:
          RuntimeError: If no attributes exist or key not found

      Example:
          >>> species = track.get_attribute("species")
      )" )
    .def(
      "has_attribute", &viame::python::track_has_attribute,
      py::arg( "key" ),
      R"(
      Check if an attribute exists.

      Args:
          key: The attribute name/key (string)

      Returns:
          True if the attribute exists, False otherwise

      Example:
          >>> if track.has_attribute("species"):
          >>>     print(track.get_attribute("species"))
      )" )
    .def(
      "attribute_keys", &viame::python::track_attribute_keys,
      R"(
      Get list of all attribute keys.

      Returns:
          List of attribute key strings

      Example:
          >>> for key in track.attribute_keys():
          >>>     print(f"{key}: {track.get_attribute(key)}")
      )" )
  ;
}
