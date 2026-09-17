// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/pipeline_framework/pipeline.h>

#if WIN32
#pragma warning (push)
#pragma warning (disable : 4244)
#endif
#include "python_wrappers.cxx"

#include <pybind11/stl_bind.h>
#if WIN32
#pragma warning (pop)
#endif

/**
 * \file pipeline.cxx
 *
 * \brief Python bindings for \link viame::pipeline::pipeline\endlink.
 */

using namespace pybind11;

namespace viame {

namespace pipeline {

namespace python {

static std::vector< wrap_port_addr > connections_from_addr(
  ::viame::pipeline::pipeline& self, ::viame::pipeline::process::name_t const& name,
  ::viame::pipeline::process::port_t const& port );
static std::vector< wrap_port_addr > receivers_for_port(
  ::viame::pipeline::pipeline& self, ::viame::pipeline::process::name_t const& name,
  ::viame::pipeline::process::port_t const& port );

} // namespace python

} // namespace pipeline

} // namespace viame

using namespace viame::pipeline::python;
PYBIND11_MODULE( pipeline, m )
{
  bind_vector< std::vector< std::string > >( m, "names_t" );

  class_< viame::pipeline::pipeline, viame::pipeline::pipeline_t >(
    m, "Pipeline",
    "A data structure for a collection of connected processes." )
    .def( init<>() )
    .def( init< viame::config_block_sptr >() )
    .def(
      "add_process", &viame::pipeline::pipeline::add_process,
      ( arg( "process" ) ),
      "Add a process to the pipeline." )
    .def(
      "remove_process", &viame::pipeline::pipeline::remove_process,
      ( arg( "name" ) ),
      "Remove a process from the pipeline." )
    .def(
      "connect", &viame::pipeline::pipeline::connect,
      arg( "upstream" ), arg( "upstream_port" ), arg( "downstream" ),
      arg( "downstream_port" ),
      "Connect two ports within the pipeline together." )
    .def(
      "disconnect", &viame::pipeline::pipeline::disconnect,
      arg( "upstream" ), arg( "upstream_port" ), arg( "downstream" ),
      arg( "downstream_port" ),
      "Disconnect two ports from each other in the pipeline." )
    .def(
      "setup_pipeline", &viame::pipeline::pipeline::setup_pipeline,
      "Prepares the pipeline for execution." )
    .def(
      "is_setup", &viame::pipeline::pipeline::is_setup,
      "Returns True if the pipeline has been setup, False otherwise." )
    .def(
      "setup_successful", &viame::pipeline::pipeline::setup_successful,
      "Returns True if the pipeline has been successfully setup, False otherwise." )
    .def(
      "reset", &viame::pipeline::pipeline::reset,
      "Resets connections and mappings within the pipeline." )
    .def(
      "reconfigure", &viame::pipeline::pipeline::reconfigure,
      ( arg( "conf" ) ),
      "Reconfigures processes within the pipeline." )
    .def(
      "process_names", &viame::pipeline::pipeline::process_names,
      "Returns a list of all process names in the pipeline." )
    .def(
      "process_by_name", &viame::pipeline::pipeline::process_by_name,
      ( arg( "name" ) ),
      "Get a process by name." )
    .def(
      "connections_from_addr", &connections_from_addr,
      arg( "name" ), arg( "port" ),
      "Return the addresses of ports that are connected downstream of a port." )
    .def(
      "connection_to_addr", &viame::pipeline::pipeline::connection_to_addr,
      arg( "name" ), arg( "port" ),
      "Return the address for the port that is connected upstream of a port." )
    .def(
      "upstream_for_process", &viame::pipeline::pipeline::upstream_for_process,
      ( arg( "name" ) ),
      "Return all processes upstream of the given process." )
    .def(
      "upstream_for_port", &viame::pipeline::pipeline::upstream_for_port,
      arg( "name" ), arg( "port" ),
      "Return the process upstream of the given port." )
    .def(
      "downstream_for_process", &viame::pipeline::pipeline::downstream_for_process,
      ( arg( "name" ) ),
      "Return all processes downstream of the given process." )
    .def(
      "downstream_for_port", &viame::pipeline::pipeline::downstream_for_port,
      arg( "name" ), arg( "port" ),
      "Return the processes downstream of the given port." )
    .def(
      "sender_for_port", &viame::pipeline::pipeline::sender_for_port,
      arg( "name" ), arg( "port" ),
      "Return the port that is sending to the given port." )
    .def(
      "receivers_for_port", &receivers_for_port,
      arg( "name" ), arg( "port" ),
      "Return the ports that are receiving from the given port." )
    .def(
      "edge_for_connection", &viame::pipeline::pipeline::edge_for_connection,
      arg( "upstream_name" ), arg( "upstream_port" ), arg( "downstream_name" ),
      arg( "downstream_port" ),
      "Returns the edge for the connection." )
    .def(
      "input_edges_for_process", &viame::pipeline::pipeline::input_edges_for_process,
      ( arg( "name" ) ),
      "Return the edges that are sending to the given process." )
    .def(
      "input_edge_for_port", &viame::pipeline::pipeline::input_edge_for_port,
      arg( "name" ), arg( "port" ),
      return_value_policy::reference,
      "Return the edge that is sending to the given port." )
    .def(
      "output_edges_for_process", &viame::pipeline::pipeline::output_edges_for_process,
      ( arg( "name" ) ),
      "Return the edges that are receiving data from the given process." )
    .def(
      "output_edges_for_port", &viame::pipeline::pipeline::output_edges_for_port,
      arg( "name" ), arg( "port" ),
      "Return the edges that are receiving data from the given port." )
  ;
}

namespace viame {

namespace pipeline {

namespace python {

std::vector< wrap_port_addr >
connections_from_addr(
  ::viame::pipeline::pipeline& self,
  ::viame::pipeline::process::name_t const& name,
  ::viame::pipeline::process::port_t const& port )
{
  ::viame::pipeline::process::port_addrs_t pair_addrs =
    self.connections_from_addr( name, port );
  std::vector< wrap_port_addr > wrap_addrs;
  for( unsigned int idx = 0; idx < pair_addrs.size(); idx++ )
  {
    wrap_addrs.push_back( wrap_port_addr( pair_addrs[ idx ] ) );
  }

  return wrap_addrs;
}

std::vector< wrap_port_addr >
receivers_for_port(
  ::viame::pipeline::pipeline& self,
  ::viame::pipeline::process::name_t const& name,
  ::viame::pipeline::process::port_t const& port )
{
  ::viame::pipeline::process::port_addrs_t pair_addrs = self.receivers_for_port(
    name,
    port );
  std::vector< wrap_port_addr > wrap_addrs;
  for( unsigned int idx = 0; idx < pair_addrs.size(); idx++ )
  {
    wrap_addrs.push_back( wrap_port_addr( pair_addrs[ idx ] ) );
  }

  return wrap_addrs;
}

} // namespace python

} // namespace pipeline

} // namespace viame
