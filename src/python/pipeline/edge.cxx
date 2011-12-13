/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/edge_exception.h>

#include <boost/python.hpp>

/**
 * \file edge.cxx
 *
 * \brief Python bindings for \link vistk::edge\endlink.
 */

using namespace boost::python;

static void translator(vistk::edge_exception const& e);

BOOST_PYTHON_MODULE(edge)
{
  register_exception_translator<
    vistk::edge_exception>(translator);

  class_<vistk::edge, vistk::edge_t, boost::noncopyable>("Edge", no_init)
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

void
translator(vistk::edge_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}
