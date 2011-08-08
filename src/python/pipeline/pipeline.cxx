/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/pipeline_exception.h>

#include <boost/python.hpp>

/**
 * \file pipeline.cxx
 *
 * \brief Python bindings for \link vistk::pipeline\endlink.
 */

using namespace boost::python;

static void translator(vistk::pipeline_exception const& e);

BOOST_PYTHON_MODULE(pipeline)
{
  register_exception_translator<
    vistk::pipeline_exception>(translator);

  class_<vistk::pipeline, vistk::pipeline_t, boost::noncopyable>("Pipeline", no_init)
    .def(init<vistk::config_t>())
    .def("add_process", &vistk::pipeline::add_process)
    .def("add_group", &vistk::pipeline::add_group)
    .def("connect", &vistk::pipeline::connect)
    .def("map_input_port", &vistk::pipeline::map_input_port)
    .def("map_output_port", &vistk::pipeline::map_output_port)
    .def("setup_pipeline", &vistk::pipeline::setup_pipeline)
    .def("process_names", &vistk::pipeline::process_names)
    .def("process_by_name", &vistk::pipeline::process_by_name)
    .def("upstream_for_process", &vistk::pipeline::upstream_for_process)
    .def("upstream_for_port", &vistk::pipeline::upstream_for_port)
    .def("downstream_for_process", &vistk::pipeline::downstream_for_process)
    .def("downstream_for_port", &vistk::pipeline::downstream_for_port)
    .def("sender_for_port", &vistk::pipeline::sender_for_port)
    .def("receivers_for_port", &vistk::pipeline::receivers_for_port)
    .def("input_edges_for_process", &vistk::pipeline::input_edges_for_process)
    .def("input_edge_for_port", &vistk::pipeline::input_edge_for_port)
    .def("output_edges_for_process", &vistk::pipeline::output_edges_for_process)
    .def("output_edges_for_port", &vistk::pipeline::output_edges_for_port)
    .def("groups", &vistk::pipeline::groups)
    .def("input_ports_for_group", &vistk::pipeline::input_ports_for_group)
    .def("output_ports_for_group", &vistk::pipeline::output_ports_for_group)
    .def("mapped_group_input_port_flags", &vistk::pipeline::mapped_group_input_port_flags)
    .def("mapped_group_output_port_flags", &vistk::pipeline::mapped_group_output_port_flags)
    .def("mapped_group_input_ports", &vistk::pipeline::mapped_group_input_ports)
    .def("mapped_group_output_port", &vistk::pipeline::mapped_group_output_port)
  ;
}

void
translator(vistk::pipeline_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}
