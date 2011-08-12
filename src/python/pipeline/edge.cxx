/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/edge_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

/**
 * \file edge.cxx
 *
 * \brief Python bindings for \link vistk::edge\endlink.
 */

using namespace boost::python;

static vistk::datum_t datum_from_edge(vistk::edge_datum_t const& edatum);
static vistk::stamp_t stamp_from_edge(vistk::edge_datum_t const& edatum);

static void translator(vistk::edge_exception const& e);

BOOST_PYTHON_MODULE(edge)
{
  register_exception_translator<
    vistk::edge_exception>(translator);

  class_<vistk::edge_datum_t>("EdgeDatum"
    , no_init)
    .def(init<vistk::datum_t, vistk::stamp_t>())
    .def("datum", &datum_from_edge)
    .def("stamp", &stamp_from_edge)
  ;
  class_<vistk::edge_data_t>("EdgeData")
    .def(vector_indexing_suite<vistk::edge_data_t>())
  ;
  class_<vistk::edges_t>("Edges")
    .def(vector_indexing_suite<vistk::edges_t>())
  ;

  class_<vistk::edge, vistk::edge_t, boost::noncopyable>("Edge"
    , no_init)
    .def(init<vistk::config_t>())
    .def("makes_dependency", &vistk::edge::makes_dependency)
    .def("has_data", &vistk::edge::has_data)
    .def("full_of_data", &vistk::edge::full_of_data)
    .def("datum_count", &vistk::edge::datum_count)
    .def("push_datum", &vistk::edge::push_datum)
    .def("get_datum", &vistk::edge::get_datum)
    .def("peek_datum", &vistk::edge::peek_datum)
    .def("pop_datum", &vistk::edge::pop_datum)
    .def("set_required_by_downstream", &vistk::edge::set_required_by_downstream)
    .def("required_by_downstream", &vistk::edge::required_by_downstream)
    .def("set_upstream_process", &vistk::edge::set_upstream_process)
    .def("set_downstream_process", &vistk::edge::set_downstream_process)
  ;
}

vistk::datum_t
datum_from_edge(vistk::edge_datum_t const& edatum)
{
  return edatum.get<0>();
}

vistk::stamp_t
stamp_from_edge(vistk::edge_datum_t const& edatum)
{
  return edatum.get<1>();
}

void
translator(vistk::edge_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}
