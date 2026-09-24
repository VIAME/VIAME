// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <typeinfo>
#include <viame/pipeline_framework/adapters/adapter_data_set.h>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

// Type conversions
#include <viame/core_types/database_query.h>
#include <viame/core_types/descriptor_request.h>
#include <viame/core_types/descriptor_set.h>
#include <viame/core_types/iqr_feedback.h>
#include <viame/core_types/track_descriptor_set.h>
#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/feature_track_set.h>
#include <viame/core_types/geo_polygon.h>
#include <viame/core_types/homography_f2f.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/object_track_set.h>
#include <viame/core_types/timestamp.h>
#include <viame/core_types/track_set.h>

#include <memory>

PYBIND11_MAKE_OPAQUE( std::vector< unsigned char > );
PYBIND11_MAKE_OPAQUE( std::vector< double > );
PYBIND11_MAKE_OPAQUE( std::vector< std::string > );

namespace ka = ::viame::adapter;
namespace py = pybind11;

namespace viame {

namespace pipeline {

namespace python {

// Accept a generic python object and cast to correct type before adding.
// This keeps a C++ process from having to deal with a py::object.
// We'll also accept datums directly so users can always use the index operator.
// When these are updated to add more types, the same will need to be done for
// datum
void
add_value_correct_type(
  ka::adapter_data_set& self,
  ::viame::pipeline::process::port_t const& port, py::object obj )
{
  if( obj.is_none() )
  {
    throw py::type_error( "Cannot add NoneType to adapter_data_set" );
  }

  if( py::isinstance< ::viame::pipeline::datum >( obj ) )
  {
    ::viame::pipeline::datum_t casted_obj = obj.cast< ::viame::pipeline::datum_t >();
    self.add_datum( port, casted_obj );
    return;
  }

#define ADS_ADD_OBJECT( PYTYPE, TYPE )        \
if( py::isinstance< PYTYPE >( obj ) )         \
{                                             \
  TYPE casted_obj = obj.cast< TYPE >();       \
  self.add_value< TYPE >( port, casted_obj ); \
  return;                                     \
}

  ADS_ADD_OBJECT( py::int_, int )
  ADS_ADD_OBJECT( py::float_, float )
  ADS_ADD_OBJECT( py::str, std::string )
  ADS_ADD_OBJECT(
    viame::image_container,
    std::shared_ptr< viame::image_container > )
  ADS_ADD_OBJECT(
    viame::descriptor_set,
    std::shared_ptr< viame::descriptor_set > )
  ADS_ADD_OBJECT(
    viame::detected_object_set,
    std::shared_ptr< viame::detected_object_set > )
  ADS_ADD_OBJECT(
    viame::track_set,
    std::shared_ptr< viame::track_set > )
  ADS_ADD_OBJECT(
    viame::feature_track_set,
    std::shared_ptr< viame::feature_track_set > )
  ADS_ADD_OBJECT(
    viame::object_track_set,
    std::shared_ptr< viame::object_track_set > )
  ADS_ADD_OBJECT(
    std::vector< double >,
    std::shared_ptr< std::vector< double > > )
  ADS_ADD_OBJECT(
    std::vector< std::string >,
    std::shared_ptr< std::vector< std::string > > )
  ADS_ADD_OBJECT(
    std::vector< unsigned char >,
    std::shared_ptr< std::vector< unsigned char > > )
  ADS_ADD_OBJECT( viame::bounding_box_d, viame::bounding_box_d )
  ADS_ADD_OBJECT( viame::timestamp, viame::timestamp )
  ADS_ADD_OBJECT( viame::geo_polygon, viame::geo_polygon )
  ADS_ADD_OBJECT( viame::f2f_homography, viame::f2f_homography )

#undef ADS_ADD_OBJECT

  throw py::type_error( "Unable to add object to adapter data set" );
}

// Take data of an unknown type from a port and return. Can't return as an "any"
// object,
// so need to cast.
//
// The 'any.type() == typeid(TYPE)' call essentially does a string comparison
// of the underlying C++ types. This allows for comparisons across pybind
// modules.
// For example, adding a datum containing a datum.VectorDouble to an ADS then
// immediately
// retrieving the value at that port will return an
// adapter_data_set.VectorDouble. The Python
// types are different, but the C++ types are the same. Note that adding a
// datum.VectorDouble
// directly will not work, due to how Pybind/Python handles opaque types. Add an
// adapter_data_set.VectorDouble instead, or add a datum containing a
// datum.VectorDouble.
// Comparing type_info hashes will not work if the types are defined in
// different modules.
// See this issue for more information:
// https://github.com/pybind/pybind11/issues/912
py::object
get_port_data_correct_type(
  ka::adapter_data_set& self,
  ::viame::pipeline::process::port_t const& port )
{
  viame::any const any =
    self.get_port_data< viame::any >( port );

#define ADS_GET_OBJECT( TYPE )                               \
if( any.type() == typeid( TYPE ) )                           \
{                                                            \
  return py::cast( viame::any_cast< TYPE >( any ) ); \
}

  ADS_GET_OBJECT( int )
  ADS_GET_OBJECT( float )
  ADS_GET_OBJECT( std::string )
  ADS_GET_OBJECT( std::shared_ptr< viame::image_container > )
  ADS_GET_OBJECT( std::shared_ptr< viame::descriptor_set > )
  ADS_GET_OBJECT( std::shared_ptr< viame::detected_object_set > )
  ADS_GET_OBJECT( std::shared_ptr< viame::track_set > )
  ADS_GET_OBJECT( std::shared_ptr< viame::feature_track_set > )
  ADS_GET_OBJECT( std::shared_ptr< viame::object_track_set > )
  ADS_GET_OBJECT( std::shared_ptr< std::vector< double > > )
  ADS_GET_OBJECT( std::shared_ptr< std::vector< std::string > > )
  ADS_GET_OBJECT( std::shared_ptr< std::vector< unsigned char > > )
  ADS_GET_OBJECT( viame::bounding_box_d )
  ADS_GET_OBJECT( viame::timestamp )
  ADS_GET_OBJECT( viame::geo_polygon )
  ADS_GET_OBJECT( viame::f2f_homography )

#undef ADS_GET_OBJECT

  std::string msg(
    "Unable to convert object found at adapter data set port: " );
  msg += port;
  msg += ". Data is of type: ";
  msg += any.type().name();
  throw py::type_error( msg );
}

// Place a typed null shared_ptr on a port. A pipeline built around input and
// output adapters expects every input port populated each step, and the ports
// it is not using want an empty sptr of the right static type.
// `add_value_correct_type` cannot express that: None carries no type, and it
// rejects None for exactly that reason.
void
add_nullptr(
  ka::adapter_data_set& self, ::viame::pipeline::process::port_t const& port,
  std::string const& type_name )
{
  if( type_name == "descriptor_request" )
  {
    self.add_value< std::shared_ptr< ::viame::descriptor_request > >(
      port, nullptr );
    return;
  }
  if( type_name == "database_query" )
  {
    self.add_value< std::shared_ptr< ::viame::database_query > >(
      port, nullptr );
    return;
  }
  if( type_name == "iqr_feedback" )
  {
    self.add_value< std::shared_ptr< ::viame::iqr_feedback > >(
      port, nullptr );
    return;
  }
  if( type_name == "uchar_vector" )
  {
    self.add_value< std::shared_ptr< std::vector< unsigned char > > >(
      port, nullptr );
    return;
  }
  if( type_name == "track_descriptor_set" )
  {
    self.add_value< ::viame::track_descriptor_set_sptr >( port, nullptr );
    return;
  }
  throw py::value_error( "add_nullptr: unsupported type name: " + type_name );
}

} // namespace python

} // namespace pipeline

} // namespace viame

PYBIND11_MODULE( adapter_data_set, m )
{
  // Here we're essentially creating bindings for vectors of different types.
  // Using pybind's automatic conversion of py::list <-> vector would cause
  // issues.
  // For example, how should the list [5, 2] be interpreted? As a vector of
  // doubles,
  // or as a vector of unsigned chars? To avoid this ambiguity, we're going to
  // bind
  // instances of vectors with certain template types explicitly.
  // If the above list is to be interpreted as a vector of doubles,
  // we would now use adapter_data_set.VectorDouble([5, 2]) on the Python side.
  py::bind_vector< std::vector< unsigned char >,
    std::shared_ptr< std::vector< unsigned char > > >(
    m, "VectorUChar",
    py::module_local( true ) );
  py::bind_vector< std::vector< double >,
    std::shared_ptr< std::vector< double > > >(
    m, "VectorDouble",
    py::module_local( true ) );
  py::bind_vector< std::vector< std::string >,
    std::shared_ptr< std::vector< std::string > > >(
    m, "VectorString",
    py::module_local( true ) );

  py::enum_< ka::adapter_data_set::data_set_type >(
    m, "DataSetType",
    "Type of data set." )
    .value( "data", ka::adapter_data_set::data )
    .value( "end_of_input", ka::adapter_data_set::end_of_input )
  ;

  py::class_< ka::adapter_data_set,
    std::shared_ptr< ka::adapter_data_set > > ads( m, "AdapterDataSet" );
  ads.def_static(
    "create", &ka::adapter_data_set::create,
    ( py::arg( "type" ) = ka::adapter_data_set::data_set_type::data ) )

    .def(
      "__iter__", [](ka::adapter_data_set& self){
        return py::make_iterator( self.cbegin(), self.cend() );
      }, py::keep_alive< 0, 1 >() )

    // Members
    .def( "type", &ka::adapter_data_set::type )
    .def( "is_end_of_data", &ka::adapter_data_set::is_end_of_data )

    // General add_value which adds any type, and __setitem__
    .def(
      "add_value", &viame::pipeline::python::add_value_correct_type,
      "This method is equivalent to using __setitem__" )
    .def( "__setitem__", &viame::pipeline::python::add_value_correct_type )

    .def( "add_datum", &ka::adapter_data_set::add_datum )

    .def( "add_nullptr", &viame::pipeline::python::add_nullptr,
          "Place a typed null on a port, for the input ports a step is "
          "not populating." )

    // General get_value which gets data of any type from a port and __getitem__
    .def(
      "get_port_data", &viame::pipeline::python::get_port_data_correct_type,
      "This method is equivalent to using __getitem__" )
    .def( "__getitem__", &viame::pipeline::python::get_port_data_correct_type )

    // The add_value function is templated.
    // To bind the function, we must bind explicit instances of it,
    // each with a different type.
    // First the native C++ types
    .def( "_add_int", &ka::adapter_data_set::add_value< int > )
    .def( "_add_float", &ka::adapter_data_set::add_value< float > )
    .def( "_add_string", &ka::adapter_data_set::add_value< std::string > )
    // Next shared ptrs to kwiver viame_algorithm_framework types
    .def(
      "_add_image_container",
      &ka::adapter_data_set::add_value< std::shared_ptr< viame::
        image_container > >,
      py::arg( "port" ), py::arg( "val" ).none( false ) )
    .def(
      "_add_descriptor_set",
      &ka::adapter_data_set::add_value< std::shared_ptr< viame::
        descriptor_set > >,
      py::arg( "port" ), py::arg( "val" ).none( false ) )
    .def(
      "_add_detected_object_set",
      &ka::adapter_data_set::add_value< std::shared_ptr< viame::
        detected_object_set > >,
      py::arg( "port" ), py::arg( "val" ).none( false ) )
    .def(
      "_add_track_set",
      &ka::adapter_data_set::add_value< std::shared_ptr< viame::
        track_set > >,
      py::arg( "port" ), py::arg( "val" ).none( false ) )
    .def(
      "_add_feature_track_set",
      &ka::adapter_data_set::add_value< std::shared_ptr< viame::
        feature_track_set > >,
      py::arg( "port" ), py::arg( "val" ).none( false ) )
    .def(
      "_add_object_track_set",
      &ka::adapter_data_set::add_value< std::shared_ptr< viame::
        object_track_set > >,
      py::arg( "port" ), py::arg( "val" ).none( false ) )
    // Next shared ptrs to native C++ types
    .def(
      "_add_double_vector",
      &ka::adapter_data_set::add_value< std::shared_ptr< std::vector< double > > >,
      py::arg( "port" ), py::arg( "val" ).none( false ) )
    .def(
      "_add_string_vector",
      &ka::adapter_data_set::add_value< std::shared_ptr< std::vector< std::
        string > > >,
      py::arg( "port" ), py::arg( "val" ).none( false ) )
    .def(
      "_add_uchar_vector",
      &ka::adapter_data_set::add_value< std::shared_ptr< std::vector< unsigned
        char > > >,
      py::arg( "port" ), py::arg( "val" ).none( false ) )
    // Next kwiver viame_algorithm_framework types
    .def(
      "_add_bounding_box",
      &ka::adapter_data_set::add_value< viame::bounding_box_d > )
    .def(
      "_add_timestamp",
      &ka::adapter_data_set::add_value< viame::timestamp > )
    .def(
      "_add_corner_points",
      &ka::adapter_data_set::add_value< viame::geo_polygon > )
    .def(
      "_add_f2f_homography",
      &ka::adapter_data_set::add_value< viame::f2f_homography > )

    .def( "empty", &ka::adapter_data_set::empty )

    // get_port_data is also templated
    .def( "_get_port_data_int", &ka::adapter_data_set::get_port_data< int > )
    .def(
      "_get_port_data_float",
      &ka::adapter_data_set::get_port_data< float > )
    .def(
      "_get_port_data_string",
      &ka::adapter_data_set::get_port_data< std::string > )
    // Next shared ptrs to kwiver viame_algorithm_framework types
    .def(
      "_get_port_data_image_container",
      &ka::adapter_data_set::get_port_data< std::shared_ptr< viame::
        image_container > > )
    .def(
      "_get_port_data_descriptor_set",
      &ka::adapter_data_set::get_port_data< std::shared_ptr< viame::
        descriptor_set > > )
    .def(
      "_get_port_data_detected_object_set",
      &ka::adapter_data_set::get_port_data< std::shared_ptr< viame::
        detected_object_set > > )
    .def(
      "_get_port_data_track_set",
      &ka::adapter_data_set::get_port_data< std::shared_ptr< viame::
        track_set > > )
    .def(
      "_get_port_data_feature_track_set",
      &ka::adapter_data_set::get_port_data< std::shared_ptr< viame::
        feature_track_set > > )
    .def(
      "_get_port_data_object_track_set",
      &ka::adapter_data_set::get_port_data< std::shared_ptr< viame::
        object_track_set > > )
    // Next shared ptrs to native C++ types
    .def(
      "_get_port_data_double_vector",
      &ka::adapter_data_set::get_port_data< std::shared_ptr< std::vector< double > > > )
    .def(
      "_get_port_data_string_vector",
      &ka::adapter_data_set::get_port_data< std::shared_ptr< std::vector< std::
        string > > > )
    .def(
      "_get_port_data_uchar_vector",
      &ka::adapter_data_set::get_port_data< std::shared_ptr< std::vector<
        unsigned char > > > )
    // Next kwiver viame_algorithm_framework types
    .def(
      "_get_port_data_bounding_box",
      &ka::adapter_data_set::get_port_data< viame::bounding_box_d > )
    .def(
      "_get_port_data_timestamp",
      &ka::adapter_data_set::get_port_data< viame::timestamp > )
    .def(
      "_get_port_data_corner_points",
      &ka::adapter_data_set::get_port_data< viame::geo_polygon > )
    .def(
      "_get_port_data_f2f_homography",
      &ka::adapter_data_set::get_port_data< viame::f2f_homography > )
    .def(
      "__nice__", [](const ka::adapter_data_set& self) -> std::string {
        auto locals = py::dict( py::arg( "self" ) = self );
        py::exec( R"(
          retval = 'size={}'.format(len(self))
      )", py::globals(), locals );
        return locals[ "retval" ].cast< std::string >();
      } )
    .def(
      "__repr__", [](py::object& self) -> std::string {
        auto locals = py::dict( py::arg( "self" ) = self );
        py::exec(
          R"(
          classname = self.__class__.__name__
          devnice = self.__nice__()
          retval = '<%s(%s) at %s>' % (classname, devnice, hex(id(self)))
      )",
          py::globals(), locals );
        return locals[ "retval" ].cast< std::string >();
      } )
    .def(
      "__str__", [](py::object& self) -> std::string {
        auto locals = py::dict( py::arg( "self" ) = self );
        py::exec(
          R"(
        from viame.pipeline import datum

        classname = self.__class__.__name__
        devnice = self.__nice__()
        retval = '<%s(%s)>\n' % (classname, devnice)
        retval += '\t{'
        for i, (port, datum_obj) in enumerate(self):
            if i:
                retval += ', '
            retval += port
            retval += ": "
            retval += str(datum_obj.get_datum())
        retval += '}'
    )",
          py::globals(), locals );
        return locals[ "retval" ].cast< std::string >();
      } )
    .def( "__len__", &ka::adapter_data_set::size )
  ;

  ads.doc() =
    R"(
      Python bindings for viame::adapter::adapter_data_set

      Example:
          >>> from viame.adapters import adapter_data_set
          >>> # Following ads has type "data". We can add/get data to/from ports
          >>> ads = adapter_data_set.AdapterDataSet.create()
          >>> assert ads.type() == adapter_data_set.DataSetType.data
          >>> # Can add as a general python object
          >>> ads["port1"] = "a_string"
          >>> # Can also add by specifying type
          >>> ads._add_int("port2", 5)
          >>> # Get both values
          >>> assert ads["port1"] == "a_string"
          >>> assert ads._get_port_data_int("port2") == 5
      )";
}
