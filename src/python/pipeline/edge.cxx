/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/edge_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/class.hpp>
#include <boost/python/exception_translator.hpp>
#include <boost/python/module.hpp>

/**
 * \file edge.cxx
 *
 * \brief Python bindings for \link vistk::edge\endlink.
 */

using namespace boost::python;

static vistk::datum_t edge_datum_datum(vistk::edge_datum_t const& edatum);
static void edge_datum_datum_set(vistk::edge_datum_t& edatum, vistk::datum_t const& dat);
static vistk::stamp_t edge_datum_stamp(vistk::edge_datum_t const& edatum);
static void edge_datum_stamp_set(vistk::edge_datum_t& edatum, vistk::stamp_t const& st);

static void translator(vistk::edge_exception const& e);

BOOST_PYTHON_MODULE(edge)
{
  register_exception_translator<
    vistk::edge_exception>(translator);

  class_<vistk::edge_datum_t>("EdgeDatum"
    , no_init)
    .def(init<vistk::datum_t, vistk::stamp_t>())
    .add_property("datum", &edge_datum_datum, &edge_datum_datum_set
      , "The datum in the packet.")
    .add_property("stamp", &edge_datum_stamp, &edge_datum_stamp_set
      , "The stamp of the packet.")
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
      , "Returns the next datum packet from the edge.")
    .def("pop_datum", &vistk::edge::pop_datum
      , "Remove the next datum packet from the edge.")
    .def("set_required_by_downstream", &vistk::edge::set_required_by_downstream
      , (arg("required"))
      , "Set whether the data within the edge is required by downstream to work.")
    .def("required_by_downstream", &vistk::edge::required_by_downstream
      , "Returns True if the downstream process needs the data in the edge.")
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
  ;
}

vistk::datum_t
edge_datum_datum(vistk::edge_datum_t const& edatum)
{
  return edatum.get<0>();
}

void
edge_datum_datum_set(vistk::edge_datum_t& edatum, vistk::datum_t const& dat)
{
  boost::get<0>(edatum) = dat;
}

vistk::stamp_t
edge_datum_stamp(vistk::edge_datum_t const& edatum)
{
  return edatum.get<1>();
}

void
edge_datum_stamp_set(vistk::edge_datum_t& edatum, vistk::stamp_t const& st)
{
  boost::get<1>(edatum) = st;
}

void
translator(vistk::edge_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}
