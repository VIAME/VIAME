// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/pipeline_framework/load_pipe_exception.h>
#include <viame/pipeline_framework/pipe_declaration_types.h>
#include <viame/pipeline_framework/pipeline_builder.h>

#include <viame/pipeline_framework/process.h>

#include <python/kwiver/sprokit/util/pystream.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <python/kwiver/sprokit/pipeline/python_wrappers.cxx>

#include <string>

/**
 * \file load.cxx
 *
 * \brief Python bindings for loading pipe blocks.
 */

using namespace pybind11;

namespace kwiver {

namespace sprokit {

namespace python {

static object pipe_block_config( ::sprokit::pipe_block const& block );
static void pipe_block_config_set(
  ::sprokit::pipe_block& block,
  ::sprokit::config_pipe_block const& config );
static object pipe_block_process( ::sprokit::pipe_block const& block );
static void pipe_block_process_set(
  ::sprokit::pipe_block& block,
  ::sprokit::process_pipe_block const& process );
static object pipe_block_connect( ::sprokit::pipe_block const& block );
static void pipe_block_connect_set(
  ::sprokit::pipe_block& block,
  ::sprokit::connect_pipe_block const& connect );
static ::sprokit::pipe_blocks load_pipe_file( std::string const& path );
static ::sprokit::pipe_blocks load_pipe( object const& stream );
} // namespace python

} // namespace sprokit

} // namespace kwiver

using namespace kwiver::sprokit::python;
PYBIND11_MODULE( load, m )
{
  bind_vector< sprokit::config_flags_t >(
    m, "ConfigFlags",
    "A collection of flags on a configuration setting." )
  ;
  class_< sprokit::config_value_t >(
    m, "ConfigValue",
    "A complete configuration setting." )
    .def( init<>() )
    .def_readwrite( "key", &sprokit::config_value_t::key_path )
    .def_readwrite( "flags", &sprokit::config_value_t::flags )
    .def_readwrite( "value", &sprokit::config_value_t::value )
  ;
  bind_vector< sprokit::config_values_t >(
    m, "ConfigValues",
    "A collection of configuration settings." )
  ;
  class_< sprokit::config_pipe_block >(
    m, "ConfigBlock",
    "A block of configuration settings." )
    .def( init<>() )
    .def_readwrite( "key", &sprokit::config_pipe_block::key )
    .def_readwrite( "values", &sprokit::config_pipe_block::values )
  ;
  class_< sprokit::process_pipe_block >(
    m, "ProcessBlock",
    "A block which declares a process." )
    .def( init<>() )
    .def_readwrite( "name", &sprokit::process_pipe_block::name )
    .def_readwrite( "type", &sprokit::process_pipe_block::type )
    .def_readwrite(
      "config_values",
      &sprokit::process_pipe_block::config_values )
  ;
  class_< sprokit::connect_pipe_block >(
    m, "ConnectBlock",
    "A block which connects two ports together." )
    .def( init<>() )
    .def_readwrite( "from_", &sprokit::connect_pipe_block::from )
    .def_readwrite( "to", &sprokit::connect_pipe_block::to )
  ;
  class_< sprokit::pipe_block >(
    m, "PipeBlock",
    "A block in a pipeline declaration file." )
    .def( init<>() )
    .def_property( "config", &pipe_block_config, &pipe_block_config_set )
    .def_property( "process", &pipe_block_process, &pipe_block_process_set )
    .def_property( "connect", &pipe_block_connect, &pipe_block_connect_set )
  ;
  class_< sprokit::pipe_blocks >(
    m, "PipeBlocks",
    "A collection of pipeline blocks." )
    .def( init<>() )
  /// \todo Need operator == on pipe_block.
  // .def(vector_indexing_suite<sprokit::pipe_blocks>())
  ;
  class_< wrap_port_addr >(
    m, "PortAddr",
    module_local(),
    "An address for a port within a pipeline." )
    .def( init<>() )
    .def_readwrite( "process", &wrap_port_addr::process )
    .def_readwrite( "port", &wrap_port_addr::port )
    .def( "getAddr", &wrap_port_addr::get_addr )
  ;
  bind_vector< std::vector< wrap_port_addr > >(
    m, "PortAddrs",
    module_local(),
    "A collection of port addresses." )
  ;

  m.def(
    "load_pipe_file", &load_pipe_file,
    call_guard< pybind11::gil_scoped_release >(),
    ( arg( "path" ) ),
    "Load pipe blocks from a file." );
  m.def(
    "load_pipe", &load_pipe, call_guard< pybind11::gil_scoped_release >(),
    ( arg( "stream" ) ),
    "Load pipe blocks from a stream." );
}

namespace kwiver {

namespace sprokit {

namespace python {

class pipe_block_visitor
{
public:
  typedef enum
  {
    BLOCK_CONFIG,
    BLOCK_PROCESS,
    BLOCK_CONNECT,
  } block_t;

  pipe_block_visitor( block_t type );
  ~pipe_block_visitor();

  block_t const block_type;

  object operator()( ::sprokit::config_pipe_block const& config_block ) const;
  object operator()( ::sprokit::process_pipe_block const& process_block ) const;
  object operator()( ::sprokit::connect_pipe_block const& connect_block ) const;
};

// ----------------------------------------------------------------------------
object
pipe_block_config( ::sprokit::pipe_block const& block )
{
  return std::visit(
    pipe_block_visitor( pipe_block_visitor::BLOCK_CONFIG ),
    block );
}

// ----------------------------------------------------------------------------
void
pipe_block_config_set(
  ::sprokit::pipe_block& block,
  ::sprokit::config_pipe_block const& config )
{
  block = config;
}

// ----------------------------------------------------------------------------
object
pipe_block_process( ::sprokit::pipe_block const& block )
{
  return std::visit(
    pipe_block_visitor( pipe_block_visitor::BLOCK_PROCESS ),
    block );
}

// ----------------------------------------------------------------------------
void
pipe_block_process_set(
  ::sprokit::pipe_block& block,
  ::sprokit::process_pipe_block const& process )
{
  block = process;
}

// ----------------------------------------------------------------------------
object
pipe_block_connect( ::sprokit::pipe_block const& block )
{
  return std::visit(
    pipe_block_visitor( pipe_block_visitor::BLOCK_CONNECT ),
    block );
}

// ----------------------------------------------------------------------------
void
pipe_block_connect_set(
  ::sprokit::pipe_block& block,
  ::sprokit::connect_pipe_block const& connect )
{
  block = connect;
}

// ----------------------------------------------------------------------------
::sprokit::pipe_blocks
load_pipe_file( std::string const& path )
{
  ::sprokit::pipeline_builder builder;
  builder.load_pipeline( path );
  return builder.pipeline_blocks();
}

// ----------------------------------------------------------------------------
::sprokit::pipe_blocks
load_pipe( object const& stream )
{
  ::sprokit::python::pyistream istr( stream );
  ::sprokit::pipeline_builder builder;
  builder.load_pipeline( istr );
  return builder.pipeline_blocks();
}

// ----------------------------------------------------------------------------
pipe_block_visitor
::pipe_block_visitor( block_t type )
  : block_type( type )
{}

// ----------------------------------------------------------------------------
pipe_block_visitor
::~pipe_block_visitor()
{}

// ----------------------------------------------------------------------------
object
pipe_block_visitor
::operator()( ::sprokit::config_pipe_block const& config_block ) const
{
  pybind11::gil_scoped_acquire acquire;
  ( void ) acquire;

  object obj = none();

  if( block_type == BLOCK_CONFIG )
  {
    obj = cast( config_block );
  }

  return obj;
}

// ----------------------------------------------------------------------------
object
pipe_block_visitor
::operator()( ::sprokit::process_pipe_block const& process_block ) const
{
  pybind11::gil_scoped_acquire acquire;
  ( void ) acquire;

  object obj = none();

  if( block_type == BLOCK_PROCESS )
  {
    obj = cast( process_block );
  }

  return obj;
}

// ----------------------------------------------------------------------------
object
pipe_block_visitor
::operator()( ::sprokit::connect_pipe_block const& connect_block ) const
{
  pybind11::gil_scoped_acquire acquire;
  ( void ) acquire;

  object obj = none();

  if( block_type == BLOCK_CONNECT )
  {
    obj = cast( connect_block );
  }

  return obj;
}

} // namespace python

} // namespace sprokit

} // namespace kwiver
