/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process_cluster.h>

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
 * \brief Python bindings for \link sprokit::pipeline\endlink.
 */

using namespace pybind11;

static std::shared_ptr<wrap_process_cluster> cluster_by_name(sprokit::pipeline& self, sprokit::process::name_t const& name);
static std::vector<wrap_port_addr> connections_from_addr(sprokit::pipeline& self, sprokit::process::name_t const& name, sprokit::process::port_t const& port);
static std::vector<wrap_port_addr> receivers_for_port(sprokit::pipeline& self, sprokit::process::name_t const& name, sprokit::process::port_t const& port);

PYBIND11_MODULE(pipeline,m)
{
  bind_vector<std::vector<std::string> >(m, "names_t");

  class_<sprokit::pipeline, sprokit::pipeline_t>(m, "Pipeline"
    , "A data structure for a collection of connected processes.")
    .def(init<>())
    .def(init<kwiver::vital::config_block_sptr>())
    .def("add_process", &sprokit::pipeline::add_process
      , (arg("process"))
      , "Add a process to the pipeline.")
    .def("remove_process", &sprokit::pipeline::remove_process
      , (arg("name"))
      , "Remove a process from the pipeline.")
    .def("connect", &sprokit::pipeline::connect
      , arg("upstream"), arg("upstream_port"), arg("downstream"), arg("downstream_port")
      , "Connect two ports within the pipeline together.")
    .def("disconnect", &sprokit::pipeline::disconnect
      , arg("upstream"), arg("upstream_port"), arg("downstream"), arg("downstream_port")
      , "Disconnect two ports from each other in the pipeline.")
    .def("setup_pipeline", &sprokit::pipeline::setup_pipeline
      , "Prepares the pipeline for execution.")
    .def("is_setup", &sprokit::pipeline::is_setup
      , "Returns True if the pipeline has been setup, False otherwise.")
    .def("setup_successful", &sprokit::pipeline::setup_successful
      , "Returns True if the pipeline has been successfully setup, False otherwise.")
    .def("reset", &sprokit::pipeline::reset
      , "Resets connections and mappings within the pipeline.")
    .def("reconfigure", &sprokit::pipeline::reconfigure
      , (arg("conf"))
      , "Reconfigures processes within the pipeline.")
    .def("process_names", &sprokit::pipeline::process_names
      , "Returns a list of all process names in the pipeline.")
    .def("process_by_name", &sprokit::pipeline::process_by_name
      , (arg("name"))
      , "Get a process by name.")
    .def("parent_cluster", &sprokit::pipeline::parent_cluster
      , (arg("name"))
      , "Get a process' parent cluster.")
    .def("cluster_names", &sprokit::pipeline::cluster_names
      , "Returns a list of all cluster names in the pipeline.")
    .def("cluster_by_name", &cluster_by_name
      , (arg("name"))
      , "Get a cluster by name.")
    .def("connections_from_addr", &connections_from_addr
      , arg("name"), arg("port")
      , "Return the addresses of ports that are connected downstream of a port.")
    .def("connection_to_addr", &sprokit::pipeline::connection_to_addr
      , arg("name"), arg("port")
      , "Return the address for the port that is connected upstream of a port.")
    .def("upstream_for_process", &sprokit::pipeline::upstream_for_process
      , (arg("name"))
      , "Return all processes upstream of the given process.")
    .def("upstream_for_port", &sprokit::pipeline::upstream_for_port
      , arg("name"), arg("port")
      , "Return the process upstream of the given port.")
    .def("downstream_for_process", &sprokit::pipeline::downstream_for_process
      , (arg("name"))
      , "Return all processes downstream of the given process.")
    .def("downstream_for_port", &sprokit::pipeline::downstream_for_port
      , arg("name"), arg("port")
      , "Return the processes downstream of the given port.")
    .def("sender_for_port", &sprokit::pipeline::sender_for_port
      , arg("name"), arg("port")
      , "Return the port that is sending to the given port.")
    .def("receivers_for_port", &receivers_for_port
      , arg("name"), arg("port")
      , "Return the ports that are receiving from the given port.")
    .def("edge_for_connection", &sprokit::pipeline::edge_for_connection
      , arg("upstream_name"), arg("upstream_port"), arg("downstream_name"), arg("downstream_port")
      , "Returns the edge for the connection.")
    .def("input_edges_for_process", &sprokit::pipeline::input_edges_for_process
      , (arg("name"))
      , "Return the edges that are sending to the given process.")
    .def("input_edge_for_port", &sprokit::pipeline::input_edge_for_port
      , arg("name"), arg("port")
      , return_value_policy::reference
      , "Return the edge that is sending to the given port.")
    .def("output_edges_for_process", &sprokit::pipeline::output_edges_for_process
      , (arg("name"))
      , "Return the edges that are receiving data from the given process.")
    .def("output_edges_for_port", &sprokit::pipeline::output_edges_for_port
      , arg("name"), arg("port")
      , "Return the edges that are receiving data from the given port.")
  ;
}

std::shared_ptr<wrap_process_cluster>
cluster_by_name(sprokit::pipeline& self, sprokit::process::name_t const& name)
{
  return std::dynamic_pointer_cast<wrap_process_cluster> (self.cluster_by_name(name));
}

std::vector<wrap_port_addr>
connections_from_addr(sprokit::pipeline& self, sprokit::process::name_t const& name, sprokit::process::port_t const& port)
{
  sprokit::process::port_addrs_t pair_addrs = self.connections_from_addr(name, port);
  std::vector<wrap_port_addr> wrap_addrs;
  for( unsigned int idx = 0; idx < pair_addrs.size(); idx++)
  {
    wrap_addrs.push_back(wrap_port_addr(pair_addrs[idx]));
  }

  return wrap_addrs;
 }

std::vector<wrap_port_addr>
receivers_for_port(sprokit::pipeline& self, sprokit::process::name_t const&name, sprokit::process::port_t const& port)
{
  sprokit::process::port_addrs_t pair_addrs = self.receivers_for_port(name, port);
  std::vector<wrap_port_addr> wrap_addrs;
  for(   unsigned int idx = 0; idx < pair_addrs.size(); idx++)
  {
    wrap_addrs.push_back(wrap_port_addr(pair_addrs[idx]));
  }

  return wrap_addrs;
}
