// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/pipeline_framework/datum.h>
#include <viame/pipeline_framework/edge.h>
#include <viame/pipeline_framework/stamp.h>

#include <pybind11/stl_bind.h>

#include "python_wrappers.cxx"
#include "python_fold.h"

/**
 * \file edge.cxx
 *
 * \brief Python bindings for \link viame::pipeline::edge\endlink.
 */

using namespace pybind11;

namespace viame {

namespace pipeline {

namespace python {

static void push_datum( ::viame::pipeline::edge& self, wrap_edge_datum const& datum );
static wrap_edge_datum get_datum( ::viame::pipeline::edge& self );
static wrap_edge_datum peek_datum(
  ::viame::pipeline::edge& self,
  pybind11::size_t const& idx );

} // namespace python

} // namespace pipeline

} // namespace viame

using namespace viame::pipeline::python;
VIAME_PYTHON_MODULE( edge, m )
{
  class_< wrap_edge_datum >( m, "EdgeDatum" )
    .def( init<>() )
    .def( init< viame::pipeline::datum, wrap_stamp >() )
    .def_readwrite( "datum", &viame::pipeline::edge_datum_t::datum )
    .def_readwrite( "stamp", &viame::pipeline::edge_datum_t::stamp )
    .def_property(
      "datum", &wrap_edge_datum::get_datum,
      &wrap_edge_datum::set_datum )
    .def_property(
      "stamp", &wrap_edge_datum::get_stamp,
      &wrap_edge_datum::set_stamp )
  ;
  bind_vector< std::vector< wrap_edge_datum > >(
    m, "EdgeData",
    "A collection of data packets that may be passed through an edge." );

  class_< viame::pipeline::edges_t >(
    m, "Edges",
    "A collection of edges." )
    .def( pybind11::init<>() );

  class_< viame::pipeline::edge, viame::pipeline::edge_t >(
    m, "Edge",
    "A communication channel between processes." )
    .def( init<>() )
    .def( init< viame::config_block_sptr >() )
    .def(
      "makes_dependency", &viame::pipeline::edge::makes_dependency,
      "Returns True if the edge implies a dependency from downstream on upstream." )
    .def(
      "has_data", &viame::pipeline::edge::has_data,
      "Returns True if the edge contains data, False otherwise." )
    .def(
      "full_of_data", &viame::pipeline::edge::full_of_data,
      "Returns True if the edge cannot hold anymore data, False otherwise." )
    .def(
      "datum_count", &viame::pipeline::edge::datum_count,
      "Returns the number of data packets within the edge." )
    .def(
      "push_datum", &push_datum,
      ( arg( "datum" ) ),
      "Pushes a datum packet into the edge." )
    .def(
      "get_datum", &get_datum,
      "Returns the next datum packet from the edge, removing it in the process." )
    .def(
      "peek_datum", &peek_datum,
      ( arg( "index" ) = 0 ),
      "Returns the next datum packet from the edge." )
    .def(
      "pop_datum", &viame::pipeline::edge::pop_datum,
      "Remove the next datum packet from the edge." )
    .def(
      "set_upstream_process", &viame::pipeline::edge::set_upstream_process,
      ( arg( "process" ) ),
      "Set the process which is feeding data into the edge." )
    .def(
      "set_downstream_process", &viame::pipeline::edge::set_downstream_process,
      ( arg( "process" ) ),
      "Set the process which is reading data from the edge." )
    .def(
      "mark_downstream_as_complete",
      &viame::pipeline::edge::mark_downstream_as_complete,
      "Indicate that the downstream process is complete." )
    .def(
      "is_downstream_complete", &viame::pipeline::edge::is_downstream_complete,
      "Returns True if the downstream process is complete, False otherwise." )
    .def_readonly_static(
      "config_dependency",
      &viame::pipeline::edge::config_dependency )
    .def_readonly_static( "config_capacity", &viame::pipeline::edge::config_capacity )
  ;
}

namespace viame {

namespace pipeline {

namespace python {

void
push_datum( ::viame::pipeline::edge& self, wrap_edge_datum const& datum )
{
  self.push_datum( ( ::viame::pipeline::edge_datum_t ) datum );
}

wrap_edge_datum
get_datum( ::viame::pipeline::edge& self )
{
  ::viame::pipeline::edge_datum_t datum = self.get_datum();
  wrap_edge_datum datum_p( *( datum.datum ), wrap_stamp( datum.stamp ) );
  return datum_p;
}

wrap_edge_datum
peek_datum( ::viame::pipeline::edge& self, pybind11::size_t const& idx )
{
  ::viame::pipeline::edge_datum_t datum = self.peek_datum( idx );
  wrap_edge_datum datum_p( *( datum.datum ), wrap_stamp( datum.stamp ) );
  return datum_p;
}

} // namespace python

} // namespace pipeline

} // namespace viame
