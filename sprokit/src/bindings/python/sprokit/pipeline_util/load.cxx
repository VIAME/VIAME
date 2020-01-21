/*ckwg +29
 * Copyright 2011-2018 by Kitware, Inc.
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

#include <sprokit/pipeline_util/pipeline_builder.h>
#include <sprokit/pipeline_util/load_pipe_exception.h>
#include <sprokit/pipeline_util/pipe_declaration_types.h>

#include <sprokit/pipeline/process.h>

#include <vital/bindings/python/vital/util/pybind11.h>
#include <sprokit/python/util/pystream.h>

#include <pybind11/stl_bind.h>

#include <bindings/python/sprokit/pipeline/python_wrappers.cxx>

#include <string>

/**
 * \file load.cxx
 *
 * \brief Python bindings for loading pipe blocks.
 */

using namespace pybind11;

static object pipe_block_config(sprokit::pipe_block const& block);
static void pipe_block_config_set(sprokit::pipe_block& block, sprokit::config_pipe_block const& config);
static object pipe_block_process(sprokit::pipe_block const& block);
static void pipe_block_process_set(sprokit::pipe_block& block, sprokit::process_pipe_block const& process);
static object pipe_block_connect(sprokit::pipe_block const& block);
static void pipe_block_connect_set(sprokit::pipe_block& block, sprokit::connect_pipe_block const& connect);
static object cluster_subblock_config(sprokit::cluster_subblock_t const& subblock);
static void cluster_subblock_config_set(sprokit::cluster_subblock_t& subblock, sprokit::cluster_config_t const& config);
static object cluster_subblock_input(sprokit::cluster_subblock_t const& subblock);
static void cluster_subblock_input_set(sprokit::cluster_subblock_t& subblock, sprokit::cluster_input_t const& input);
static object cluster_subblock_output(sprokit::cluster_subblock_t const& subblock);
static void cluster_subblock_output_set(sprokit::cluster_subblock_t& subblock, sprokit::cluster_output_t const& output);
static object cluster_block_config(sprokit::cluster_block const& block);
static void cluster_block_config_set(sprokit::cluster_block& block, sprokit::config_pipe_block const& config);
static object cluster_block_process(sprokit::cluster_block const& block);
static void cluster_block_process_set(sprokit::cluster_block& block, sprokit::process_pipe_block const& process);
static object cluster_block_connect(sprokit::cluster_block const& block);
static void cluster_block_connect_set(sprokit::cluster_block& block, sprokit::connect_pipe_block const& connect);
static object cluster_block_cluster(sprokit::cluster_block const& block);
static void cluster_block_cluster_set(sprokit::cluster_block& block, sprokit::cluster_pipe_block const& cluster);
static sprokit::pipe_blocks load_pipe_file(std::string const& path);
static sprokit::pipe_blocks load_pipe(object const& stream);
static sprokit::cluster_blocks load_cluster_file(std::string const& path);
static sprokit::cluster_blocks load_cluster(object const& stream);
static std::vector<wrap_port_addr> get_targets(sprokit::cluster_input_t const& self);
static void set_targets(sprokit::cluster_input_t &self, std::vector<wrap_port_addr> const& wrap_targets);

PYBIND11_MODULE(load, m)
{
  bind_vector<sprokit::config_flags_t>(m, "ConfigFlags"
    , "A collection of flags on a configuration setting.")
  ;
  class_<sprokit::config_value_t>(m, "ConfigValue"
    , "A complete configuration setting.")
    .def(init<>())
    .def_readwrite("key", &sprokit::config_value_t::key_path)
    .def_readwrite("flags", &sprokit::config_value_t::flags)
    .def_readwrite("value", &sprokit::config_value_t::value)
  ;
  bind_vector<sprokit::config_values_t>(m, "ConfigValues"
    , "A collection of configuration settings.")
  ;
  class_<sprokit::config_pipe_block>(m, "ConfigBlock"
    , "A block of configuration settings.")
    .def(init<>())
    .def_readwrite("key", &sprokit::config_pipe_block::key)
    .def_readwrite("values", &sprokit::config_pipe_block::values)
  ;
  class_<sprokit::process_pipe_block>(m, "ProcessBlock"
    , "A block which declares a process.")
    .def(init<>())
    .def_readwrite("name", &sprokit::process_pipe_block::name)
    .def_readwrite("type", &sprokit::process_pipe_block::type)
    .def_readwrite("config_values", &sprokit::process_pipe_block::config_values)
  ;
  class_<sprokit::connect_pipe_block>(m, "ConnectBlock"
    , "A block which connects two ports together.")
    .def(init<>())
    .def_readwrite("from_", &sprokit::connect_pipe_block::from)
    .def_readwrite("to", &sprokit::connect_pipe_block::to)
  ;
  class_<sprokit::pipe_block>(m, "PipeBlock"
    , "A block in a pipeline declaration file.")
    .def(init<>())
    .def_property("config", &pipe_block_config, &pipe_block_config_set)
    .def_property("process", &pipe_block_process, &pipe_block_process_set)
    .def_property("connect", &pipe_block_connect, &pipe_block_connect_set)
  ;
  class_<sprokit::pipe_blocks>(m, "PipeBlocks"
    , "A collection of pipeline blocks.")
    .def(init<>())
    /// \todo Need operator == on pipe_block.
    //.def(vector_indexing_suite<sprokit::pipe_blocks>())
  ;
  class_<sprokit::cluster_config_t>(m, "ClusterConfig"
    , "A configuration value for a cluster.")
    .def(init<>())
    .def_readwrite("description", &sprokit::cluster_config_t::description)
    .def_readwrite("config_value", &sprokit::cluster_config_t::config_value)
  ;
  class_<sprokit::cluster_input_t>(m, "ClusterInput"
    , "An input mapping for a cluster.")
    .def(init<>())
    .def_readwrite("description", &sprokit::cluster_input_t::description)
    .def_readwrite("from_", &sprokit::cluster_input_t::from)
    .def_property("targets", &get_targets, &set_targets)
  ;
  class_<sprokit::cluster_output_t>(m, "ClusterOutput"
    , "An output mapping for a cluster.")
    .def(init<>())
    .def_readwrite("description", &sprokit::cluster_output_t::description)
    .def_readwrite("from_", &sprokit::cluster_output_t::from)
    .def_readwrite("to", &sprokit::cluster_output_t::to)
  ;
  class_<sprokit::cluster_subblock_t>(m, "ClusterSubblock"
    , "A subblock within a cluster.")
    .def(init<>())
    .def_property("config", &cluster_subblock_config, &cluster_subblock_config_set)
    .def_property("input", &cluster_subblock_input, &cluster_subblock_input_set)
    .def_property("output", &cluster_subblock_output, &cluster_subblock_output_set)
  ;
  class_<sprokit::cluster_subblocks_t>(m, "ClusterSubblocks"
    , "A collection of cluster subblocks.")
    .def(init<>())
    /// \todo Need operator == on cluster_subblock_t for pybind11::bind_vector
  ;
  class_<sprokit::cluster_pipe_block>(m, "ClusterBlock"
    , "A block which declares a cluster within the pipeline.")
    .def(init<>())
    .def_readwrite("type", &sprokit::cluster_pipe_block::type)
    .def_readwrite("description", &sprokit::cluster_pipe_block::description)
    .def_readwrite("subblocks", &sprokit::cluster_pipe_block::subblocks)
  ;
  class_<sprokit::cluster_block>(m, "ClusterDefineBlock"
    , "A block in a pipeline declaration file.")
    .def(init<>())
    .def_property("config", &cluster_block_config, &cluster_block_config_set)
    .def_property("process", &cluster_block_process, &cluster_block_process_set)
    .def_property("connect", &cluster_block_connect, &cluster_block_connect_set)
    .def_property("cluster", &cluster_block_cluster, &cluster_block_cluster_set)
  ;
  class_<sprokit::cluster_blocks>(m, "ClusterDefineBlocks"
    , "A collection of cluster blocks.")
    .def(init<>())
    /// \todo Need operator == on cluster_block for pybind11::bind_vector
  ;

  class_<wrap_port_addr>(m, "PortAddr"
    , module_local()
    , "An address for a port within a pipeline.")
    .def(init<>())
    .def_readwrite("process", &wrap_port_addr::process)
    .def_readwrite("port", &wrap_port_addr::port)
    .def("getAddr", &wrap_port_addr::get_addr)
  ;
  bind_vector<std::vector<wrap_port_addr> >(m, "PortAddrs"
    , module_local()
    , "A collection of port addresses.")
  ;

  m.def("load_pipe_file", &load_pipe_file, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("path"))
    , "Load pipe blocks from a file.");
  m.def("load_pipe", &load_pipe, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("stream"))
    , "Load pipe blocks from a stream.");
  m.def("load_cluster_file", &load_cluster_file, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("path"))
    , "Load cluster blocks from a file.");
  m.def("load_cluster", &load_cluster, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("stream"))
    , "Load cluster blocks from a stream.");

}

class pipe_block_visitor
{
  public:
    typedef enum
    {
      BLOCK_CONFIG,
      BLOCK_PROCESS,
      BLOCK_CONNECT,
      BLOCK_CLUSTER
    } block_t;

    pipe_block_visitor(block_t type);
    ~pipe_block_visitor();

    block_t const block_type;

    object operator () (sprokit::config_pipe_block const& config_block) const;
    object operator () (sprokit::process_pipe_block const& process_block) const;
    object operator () (sprokit::connect_pipe_block const& connect_block) const;
    object operator () (sprokit::cluster_pipe_block const& cluster_block) const;
};


// ----------------------------------------------------------------------------
object
pipe_block_config(sprokit::pipe_block const& block)
{
  return kwiver::vital::visit( pipe_block_visitor(pipe_block_visitor::BLOCK_CONFIG), block);
}


// ----------------------------------------------------------------------------
void
pipe_block_config_set(sprokit::pipe_block& block, sprokit::config_pipe_block const& config)
{
  block = config;
}


// ----------------------------------------------------------------------------
object
pipe_block_process(sprokit::pipe_block const& block)
{
  return kwiver::vital::visit(pipe_block_visitor(pipe_block_visitor::BLOCK_PROCESS), block);
}


// ----------------------------------------------------------------------------
void
pipe_block_process_set(sprokit::pipe_block& block, sprokit::process_pipe_block const& process)
{
  block = process;
}


// ----------------------------------------------------------------------------
object
pipe_block_connect(sprokit::pipe_block const& block)
{
  return kwiver::vital::visit(pipe_block_visitor(pipe_block_visitor::BLOCK_CONNECT), block);
}


// ----------------------------------------------------------------------------
void
pipe_block_connect_set(sprokit::pipe_block& block, sprokit::connect_pipe_block const& connect)
{
  block = connect;
}


// ----------------------------------------------------------------------------
class cluster_subblock_visitor
{
  public:
    typedef enum
    {
      BLOCK_CONFIG,
      BLOCK_INPUT,
      BLOCK_OUTPUT
    } block_t;

    cluster_subblock_visitor(block_t type);
    ~cluster_subblock_visitor();

    block_t const block_type;

    object operator () (sprokit::cluster_config_t const& config) const;
    object operator () (sprokit::cluster_input_t const& input) const;
    object operator () (sprokit::cluster_output_t const& output) const;
};


// ----------------------------------------------------------------------------
object
cluster_subblock_config(sprokit::cluster_subblock_t const& subblock)
{
  return kwiver::vital::visit(cluster_subblock_visitor(cluster_subblock_visitor::BLOCK_CONFIG), subblock);
}


// ----------------------------------------------------------------------------
void
cluster_subblock_config_set(sprokit::cluster_subblock_t& subblock, sprokit::cluster_config_t const& config)
{
  subblock = config;
}


// ----------------------------------------------------------------------------
object
cluster_subblock_input(sprokit::cluster_subblock_t const& subblock)
{
  return kwiver::vital::visit(cluster_subblock_visitor(cluster_subblock_visitor::BLOCK_INPUT), subblock);
}


// ----------------------------------------------------------------------------
void
cluster_subblock_input_set(sprokit::cluster_subblock_t& subblock, sprokit::cluster_input_t const& input)
{
  subblock = input;
}


// ----------------------------------------------------------------------------
object
cluster_subblock_output(sprokit::cluster_subblock_t const& subblock)
{
  return kwiver::vital::visit(cluster_subblock_visitor(cluster_subblock_visitor::BLOCK_OUTPUT), subblock);
}


// ----------------------------------------------------------------------------
void
cluster_subblock_output_set(sprokit::cluster_subblock_t& subblock, sprokit::cluster_output_t const& output)
{
  subblock = output;
}


// ----------------------------------------------------------------------------
object
cluster_block_config(sprokit::cluster_block const& block)
{
  return kwiver::vital::visit(pipe_block_visitor(pipe_block_visitor::BLOCK_CONFIG), block);
}


// ----------------------------------------------------------------------------
void
cluster_block_config_set(sprokit::cluster_block& block, sprokit::config_pipe_block const& config)
{
  block = config;
}


// ----------------------------------------------------------------------------
object
cluster_block_process(sprokit::cluster_block const& block)
{
  return kwiver::vital::visit(pipe_block_visitor(pipe_block_visitor::BLOCK_PROCESS), block);
}


// ----------------------------------------------------------------------------
void
cluster_block_process_set(sprokit::cluster_block& block, sprokit::process_pipe_block const& process)
{
  block = process;
}


// ----------------------------------------------------------------------------
object
cluster_block_connect(sprokit::cluster_block const& block)
{
  return kwiver::vital::visit(pipe_block_visitor(pipe_block_visitor::BLOCK_CONNECT), block);
}


// ----------------------------------------------------------------------------
void
cluster_block_connect_set(sprokit::cluster_block& block, sprokit::connect_pipe_block const& connect)
{
  block = connect;
}


// ----------------------------------------------------------------------------
object
cluster_block_cluster(sprokit::cluster_block const& block)
{
  return kwiver::vital::visit(pipe_block_visitor(pipe_block_visitor::BLOCK_CLUSTER), block);
}


// ----------------------------------------------------------------------------
void
cluster_block_cluster_set(sprokit::cluster_block& block, sprokit::cluster_pipe_block const& cluster)
{
  block = cluster;
}


// ----------------------------------------------------------------------------
sprokit::pipe_blocks
load_pipe_file(std::string const& path)
{
  sprokit::pipeline_builder builder;
  builder.load_pipeline( path );
  return builder.pipeline_blocks();
}


// ----------------------------------------------------------------------------
sprokit::pipe_blocks
load_pipe(object const& stream)
{
  sprokit::python::pyistream istr(stream);
  sprokit::pipeline_builder builder;
  builder.load_pipeline( istr );
  return builder.pipeline_blocks();
}


// ----------------------------------------------------------------------------
sprokit::cluster_blocks
load_cluster_file(std::string const& path)
{
  sprokit::pipeline_builder builder;
  builder.load_cluster( path );
  return builder.cluster_blocks();
}


// ----------------------------------------------------------------------------
sprokit::cluster_blocks
load_cluster(object const& stream)
{
  sprokit::python::pyistream istr(stream);
  sprokit::pipeline_builder builder;
  builder.load_cluster( istr );
  return builder.cluster_blocks();
}


// ----------------------------------------------------------------------------
pipe_block_visitor
::pipe_block_visitor(block_t type)
  : block_type(type)
{
}


// ----------------------------------------------------------------------------
pipe_block_visitor
::~pipe_block_visitor()
{
}


// ----------------------------------------------------------------------------
object
pipe_block_visitor
::operator () (sprokit::config_pipe_block const& config_block) const
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  object obj = none();

  if (block_type == BLOCK_CONFIG)
  {
    obj = cast(config_block);
  }

  return obj;
}


// ----------------------------------------------------------------------------
object
pipe_block_visitor
::operator () (sprokit::process_pipe_block const& process_block) const
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  object obj = none();

  if (block_type == BLOCK_PROCESS)
  {
    obj = cast(process_block);
  }

  return obj;
}


// ----------------------------------------------------------------------------
object
pipe_block_visitor
::operator () (sprokit::connect_pipe_block const& connect_block) const
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  object obj = none();

  if (block_type == BLOCK_CONNECT)
  {
    obj = cast(connect_block);
  }

  return obj;
}


// ----------------------------------------------------------------------------
object
pipe_block_visitor
::operator () (sprokit::cluster_pipe_block const& cluster_block) const
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  object obj = none();

  if (block_type == BLOCK_CLUSTER)
  {
    obj = cast(cluster_block);
  }

  return obj;
}


// ----------------------------------------------------------------------------
cluster_subblock_visitor
::cluster_subblock_visitor(block_t type)
  : block_type(type)
{
}


// ----------------------------------------------------------------------------
cluster_subblock_visitor
::~cluster_subblock_visitor()
{
}


// ----------------------------------------------------------------------------
object
cluster_subblock_visitor
::operator () (sprokit::cluster_config_t const& config) const
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  if (block_type == BLOCK_CONFIG)
  {
    return cast(config);
  }

  return none();
}


// ----------------------------------------------------------------------------
object
cluster_subblock_visitor
::operator () (sprokit::cluster_input_t const& input) const
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  if (block_type == BLOCK_INPUT)
  {
    return cast(input);
  }

  return none();
}


// ----------------------------------------------------------------------------
object
cluster_subblock_visitor
::operator () (sprokit::cluster_output_t const& output) const
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  if (block_type == BLOCK_OUTPUT)
  {
    return cast(output);
  }

  return none();
}


// ----------------------------------------------------------------------------
std::vector<wrap_port_addr>
get_targets(sprokit::cluster_input_t const& self)
{
  std::vector<wrap_port_addr> wrap_targets;
  for( auto target : self.targets)
  {
    wrap_targets.push_back(wrap_port_addr(target));
  }
  return wrap_targets;
}


// ----------------------------------------------------------------------------
void
set_targets(sprokit::cluster_input_t &self, std::vector<wrap_port_addr> const& wrap_targets)
{
  sprokit::process::port_addrs_t targets;
  for( auto wrap_target : wrap_targets)
  {
    targets.push_back(wrap_target.get_addr());
  }
  self.targets = targets;
}
