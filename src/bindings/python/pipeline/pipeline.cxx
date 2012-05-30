/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/pipeline_exception.h>

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>

/**
 * \file pipeline.cxx
 *
 * \brief Python bindings for \link vistk::pipeline\endlink.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(pipeline)
{
  class_<vistk::pipeline, vistk::pipeline_t, boost::noncopyable>("Pipeline"
    , "A data structure for a collection of connected processes."
    , no_init)
    .def(init<vistk::config_t>())
    .def("add_process", &vistk::pipeline::add_process
      , (arg("process"))
      , "Add a process to the pipeline.")
    .def("add_group", &vistk::pipeline::add_group
      , (arg("group"))
      , "Create a group within the pipeline.")
    .def("remove_process", &vistk::pipeline::remove_process
      , (arg("name"))
      , "Remove a process from the pipeline.")
    .def("remove_group", &vistk::pipeline::remove_group
      , (arg("group"))
      , "Remove a group from the pipeline.")
    .def("connect", &vistk::pipeline::connect
      , (arg("upstream"), arg("upstream_port"), arg("downstream"), arg("downstream_port"))
      , "Connect two ports within the pipeline together.")
    .def("disconnect", &vistk::pipeline::disconnect
      , (arg("upstream"), arg("upstream_port"), arg("downstream"), arg("downstream_port"))
      , "Disconnect two ports from each other in the pipeline.")
    .def("map_input_port", &vistk::pipeline::map_input_port
      , (arg("group"), arg("group_port"), arg("name"), arg("process_port"), arg("flags"))
      , "Maps a group input port to an input port on a process.")
    .def("map_output_port", &vistk::pipeline::map_output_port
      , (arg("group"), arg("group_port"), arg("name"), arg("process_port"), arg("flags"))
      , "Maps a group output port to an output port on a process.")
    .def("unmap_input_port", &vistk::pipeline::unmap_input_port
      , (arg("group"), arg("group_port"), arg("name"), arg("process_port"))
      , "Unmaps a group input port from an input port on a process.")
    .def("unmap_output_port", &vistk::pipeline::unmap_output_port
      , (arg("group"), arg("group_port"), arg("name"), arg("process_port"))
      , "Unmaps a group output port from an output port on a process.")
    .def("setup_pipeline", &vistk::pipeline::setup_pipeline
      , "Prepares the pipeline for execution.")
    .def("is_setup", &vistk::pipeline::is_setup
      , "Returns True if the pipeline has been setup, False otherwise.")
    .def("setup_successful", &vistk::pipeline::setup_successful
      , "Returns True if the pipeline has been successfully setup, False otherwise.")
    .def("reset", &vistk::pipeline::reset
      , "Resets connections and mappings within the pipeline.")
    .def("start", &vistk::pipeline::start
      , "Notify the pipeline that its execution has started.")
    .def("stop", &vistk::pipeline::stop
      , "Notify the pipeline that its execution has ended.")
    .def("process_names", &vistk::pipeline::process_names
      , "Returns a list of all process names in the pipeline.")
    .def("process_by_name", &vistk::pipeline::process_by_name
      , (arg("name"))
      , "Get a process by name.")
    .def("connections_from_addr", &vistk::pipeline::connections_from_addr
      , (arg("name"), arg("port"))
      , "Return the addresses of ports that are connected downstream of a port.")
    .def("connection_to_addr", &vistk::pipeline::connection_to_addr
      , (arg("name"), arg("port"))
      , "Return the address for the port that is connected upstream of a port.")
    .def("upstream_for_process", &vistk::pipeline::upstream_for_process
      , (arg("name"))
      , "Return all processes upstream of the given process.")
    .def("upstream_for_port", &vistk::pipeline::upstream_for_port
      , (arg("name"), arg("port"))
      , "Return the process upstream of the given port.")
    .def("downstream_for_process", &vistk::pipeline::downstream_for_process
      , (arg("name"))
      , "Return all processes downstream of the given process.")
    .def("downstream_for_port", &vistk::pipeline::downstream_for_port
      , (arg("name"), arg("port"))
      , "Return the processes downstream of the given port.")
    .def("sender_for_port", &vistk::pipeline::sender_for_port
      , (arg("name"), arg("port"))
      , "Return the port that is sending to the given port.")
    .def("receivers_for_port", &vistk::pipeline::receivers_for_port
      , (arg("name"), arg("port"))
      , "Return the ports that are receiving from the given port.")
    .def("edge_for_connection", &vistk::pipeline::edge_for_connection
      , (arg("upstream_name"), arg("upstream_port"), arg("downstream_name"), arg("downstream_port"))
      , "Returns the edge for the connection.")
    .def("input_edges_for_process", &vistk::pipeline::input_edges_for_process
      , (arg("name"))
      , "Return the edges that are sending to the given process.")
    .def("input_edge_for_port", &vistk::pipeline::input_edge_for_port
      , (arg("name"), arg("port"))
      , "Return the edge that is sending to the given port.")
    .def("output_edges_for_process", &vistk::pipeline::output_edges_for_process
      , (arg("name"))
      , "Return the edges that are receiving data from the given process.")
    .def("output_edges_for_port", &vistk::pipeline::output_edges_for_port
      , (arg("name"), arg("port"))
      , "Return the edges that are receiving data from the given port.")
    .def("groups", &vistk::pipeline::groups
      , "Returns a list of all groups in the pipeline.")
    .def("input_ports_for_group", &vistk::pipeline::input_ports_for_group
      , (arg("name"))
      , "Return the input ports on a group.")
    .def("output_ports_for_group", &vistk::pipeline::output_ports_for_group
      , (arg("name"))
      , "Return the output ports on a group.")
    .def("mapped_group_input_port_flags", &vistk::pipeline::mapped_group_input_port_flags
      , (arg("name"), arg("port"))
      , "Return the flags on a group\'s input port.")
    .def("mapped_group_output_port_flags", &vistk::pipeline::mapped_group_output_port_flags
      , (arg("name"), arg("port"))
      , "Return the flags on a group\'s output port.")
    .def("mapped_group_input_ports", &vistk::pipeline::mapped_group_input_ports
      , (arg("name"), arg("port"))
      , "Return the ports that are mapped to the group\'s input port.")
    .def("mapped_group_output_port", &vistk::pipeline::mapped_group_output_port
      , (arg("name"), arg("port"))
      , "Return the ports that are mapped to the group\'s output port.")
  ;
}
