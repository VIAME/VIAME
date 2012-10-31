/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/stamp.h>

#include <vistk/python/util/python_gil.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>

/**
 * \file edge.cxx
 *
 * \brief Python bindings for \link vistk::edge\endlink.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(edge)
{
  class_<vistk::edge_datum_t>("EdgeDatum"
    , no_init)
    .def(init<>())
    .def(init<vistk::datum_t, vistk::stamp_t>())
    .def_readwrite("datum", &vistk::edge_datum_t::datum)
    .def_readwrite("stamp", &vistk::edge_datum_t::stamp)
  ;
  class_<vistk::edge_data_t>("EdgeData"
    , "A collection of data packets that may be passed through an edge.")
    .def(vector_indexing_suite<vistk::edge_data_t>())
  ;
  class_<vistk::edges_t>("Edges"
    , "A collection of edges.")
    .def(vector_indexing_suite<vistk::edges_t>())
  ;

  class_<vistk::edge, vistk::edge_t, boost::noncopyable>("Edge"
    , "A communication channel between processes."
    , no_init)
    .def(init<>())
    .def(init<vistk::config_t>())
    .def("makes_dependency", &vistk::edge::makes_dependency
      , "Returns True if the edge implies a dependency from downstream on upstream.")
    .def("has_data", &vistk::edge::has_data
      , "Returns True if the edge contains data, False otherwise.")
    .def("full_of_data", &vistk::edge::full_of_data
      , "Returns True if the edge cannot hold anymore data, False otherwise.")
    .def("datum_count", &vistk::edge::datum_count
      , "Returns the number of data packets within the edge.")
    .def("push_datum", &vistk::edge::push_datum
      , (arg("datum"))
      , "Pushes a datum packet into the edge.")
    .def("get_datum", &vistk::edge::get_datum
      , "Returns the next datum packet from the edge, removing it in the process.")
    .def("peek_datum", &vistk::edge::peek_datum
      , (arg("index") = 0)
      , "Returns the next datum packet from the edge.")
    .def("pop_datum", &vistk::edge::pop_datum
      , "Remove the next datum packet from the edge.")
    .def("set_upstream_process", &vistk::edge::set_upstream_process
      , (arg("process"))
      , "Set the process which is feeding data into the edge.")
    .def("set_downstream_process", &vistk::edge::set_downstream_process
      , (arg("process"))
      , "Set the process which is reading data from the edge.")
    .def("mark_downstream_as_complete", &vistk::edge::mark_downstream_as_complete
      , "Indicate that the downstream process is complete.")
    .def("is_downstream_complete", &vistk::edge::is_downstream_complete
      , "Returns True if the downstream process is complete, False otherwise.")
    .def_readonly("config_dependency", &vistk::edge::config_dependency)
    .def_readonly("config_capacity", &vistk::edge::config_capacity)
  ;
}
