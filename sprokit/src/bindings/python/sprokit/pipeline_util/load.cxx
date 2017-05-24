/*ckwg +29
 * Copyright 2011-2012 by Kitware, Inc.
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

#include <sprokit/pipeline_util/load_pipe.h>
#include <sprokit/pipeline_util/load_pipe_exception.h>
#include <sprokit/pipeline_util/pipe_declaration_types.h>
// #include <sprokit/pipeline_util/path.h>

#include <sprokit/pipeline/process.h>

#include <sprokit/python/util/pystream.h>
#include <sprokit/python/util/python_convert_optional.h>
#include <sprokit/python/util/python_gil.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/exception_translator.hpp>

#include <string>

/**
 * \file load.cxx
 *
 * \brief Python bindings for loading pipe blocks.
 */

using namespace boost::python;

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

BOOST_PYTHON_MODULE(load)
{
  sprokit::python::register_optional_converter<sprokit::process::port_flags_t>("PortFlagsOpt", "An optional port flags.");

  class_<sprokit::token_t>("Token"
    , "A token in the pipeline description.");
  class_<sprokit::config_flag_t>("ConfigFlag"
    , "A flag on a configuration setting.");
  class_<sprokit::config_flags_t>("ConfigFlags"
    , "A collection of flags on a configuration setting.")
    .def(vector_indexing_suite<sprokit::config_flags_t>())
  ;
  class_<sprokit::config_value_t>("ConfigValue"
    , "A complete configuration setting.")
    .def_readwrite("key", &sprokit::config_value_t::key_path)
    .def_readwrite("flags", &sprokit::config_value_t::flags)
    .def_readwrite("value", &sprokit::config_value_t::value)
  ;
  class_<sprokit::config_values_t>("ConfigValues"
    , "A collection of configuration settings.")
    /// \todo Need operator == on config_value_t
    //.def(vector_indexing_suite<sprokit::config_values_t>())
  ;
  class_<sprokit::config_pipe_block>("ConfigBlock"
    , "A block of configuration settings.")
    .def_readwrite("key", &sprokit::config_pipe_block::key)
    .def_readwrite("values", &sprokit::config_pipe_block::values)
  ;
  class_<sprokit::process_pipe_block>("ProcessBlock"
    , "A block which declares a process.")
    .def_readwrite("name", &sprokit::process_pipe_block::name)
    .def_readwrite("type", &sprokit::process_pipe_block::type)
    .def_readwrite("config_values", &sprokit::process_pipe_block::config_values)
  ;
  class_<sprokit::connect_pipe_block>("ConnectBlock"
    , "A block which connects two ports together.")
    .def_readwrite("from_", &sprokit::connect_pipe_block::from)
    .def_readwrite("to", &sprokit::connect_pipe_block::to)
  ;
  class_<sprokit::pipe_block>("PipeBlock"
    , "A block in a pipeline declaration file.")
    .add_property("config", &pipe_block_config, &pipe_block_config_set)
    .add_property("process", &pipe_block_process, &pipe_block_process_set)
    .add_property("connect", &pipe_block_connect, &pipe_block_connect_set)
  ;
  class_<sprokit::pipe_blocks>("PipeBlocks"
    , "A collection of pipeline blocks.")
    /// \todo Need operator == on pipe_block.
    //.def(vector_indexing_suite<sprokit::pipe_blocks>())
  ;
  class_<sprokit::cluster_config_t>("ClusterConfig"
    , "A configuration value for a cluster.")
    .def_readwrite("description", &sprokit::cluster_config_t::description)
    .def_readwrite("config_value", &sprokit::cluster_config_t::config_value)
  ;
  class_<sprokit::cluster_input_t>("ClusterInput"
    , "An input mapping for a cluster.")
    .def_readwrite("description", &sprokit::cluster_input_t::description)
    .def_readwrite("from_", &sprokit::cluster_input_t::from)
    .def_readwrite("targets", &sprokit::cluster_input_t::targets)
  ;
  class_<sprokit::cluster_output_t>("ClusterOutput"
    , "An output mapping for a cluster.")
    .def_readwrite("description", &sprokit::cluster_output_t::description)
    .def_readwrite("from_", &sprokit::cluster_output_t::from)
    .def_readwrite("to", &sprokit::cluster_output_t::to)
  ;
  class_<sprokit::cluster_subblock_t>("ClusterSubblock"
    , "A subblock within a cluster.")
    .add_property("config", &cluster_subblock_config, &cluster_subblock_config_set)
    .add_property("input", &cluster_subblock_input, &cluster_subblock_input_set)
    .add_property("output", &cluster_subblock_output, &cluster_subblock_output_set)
  ;
  class_<sprokit::cluster_subblocks_t>("ClusterSubblocks"
    , "A collection of cluster subblocks.")
    /// \todo Need operator == on cluster_subblock_t.
    //.def(vector_indexing_suite<sprokit::cluster_subblocks_t>())
  ;
  class_<sprokit::cluster_pipe_block>("ClusterBlock"
    , "A block which declares a cluster within the pipeline.")
    .def_readwrite("type", &sprokit::cluster_pipe_block::type)
    .def_readwrite("description", &sprokit::cluster_pipe_block::description)
    .def_readwrite("subblocks", &sprokit::cluster_pipe_block::subblocks)
  ;
  class_<sprokit::cluster_block>("ClusterDefineBlock"
    , "A block in a pipeline declaration file.")
    .add_property("config", &cluster_block_config, &cluster_block_config_set)
    .add_property("process", &cluster_block_process, &cluster_block_process_set)
    .add_property("connect", &cluster_block_connect, &cluster_block_connect_set)
    .add_property("cluster", &cluster_block_cluster, &cluster_block_cluster_set)
  ;
  class_<sprokit::cluster_blocks>("ClusterDefineBlocks"
    , "A collection of cluster blocks.")
    /// \todo Need operator == on cluster_block.
    //.def(vector_indexing_suite<sprokit::cluster_blocks>())
  ;

  def("load_pipe_file", &load_pipe_file
    , (arg("path"))
    , "Load pipe blocks from a file.");
  def("load_pipe", &load_pipe
    , (arg("stream"))
    , "Load pipe blocks from a stream.");
  def("load_cluster_file", &load_cluster_file
    , (arg("path"))
    , "Load cluster blocks from a file.");
  def("load_cluster", &load_cluster
    , (arg("stream"))
    , "Load cluster blocks from a stream.");
}

class pipe_block_visitor
  : public boost::static_visitor<object>
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

object
pipe_block_config(sprokit::pipe_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_CONFIG), block);
}

void
pipe_block_config_set(sprokit::pipe_block& block, sprokit::config_pipe_block const& config)
{
  block = config;
}

object
pipe_block_process(sprokit::pipe_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_PROCESS), block);
}

void
pipe_block_process_set(sprokit::pipe_block& block, sprokit::process_pipe_block const& process)
{
  block = process;
}

object
pipe_block_connect(sprokit::pipe_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_CONNECT), block);
}

void
pipe_block_connect_set(sprokit::pipe_block& block, sprokit::connect_pipe_block const& connect)
{
  block = connect;
}

class cluster_subblock_visitor
  : public boost::static_visitor<object>
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

object
cluster_subblock_config(sprokit::cluster_subblock_t const& subblock)
{
  return boost::apply_visitor(cluster_subblock_visitor(cluster_subblock_visitor::BLOCK_CONFIG), subblock);
}

void
cluster_subblock_config_set(sprokit::cluster_subblock_t& subblock, sprokit::cluster_config_t const& config)
{
  subblock = config;
}

object
cluster_subblock_input(sprokit::cluster_subblock_t const& subblock)
{
  return boost::apply_visitor(cluster_subblock_visitor(cluster_subblock_visitor::BLOCK_INPUT), subblock);
}

void
cluster_subblock_input_set(sprokit::cluster_subblock_t& subblock, sprokit::cluster_input_t const& input)
{
  subblock = input;
}

object
cluster_subblock_output(sprokit::cluster_subblock_t const& subblock)
{
  return boost::apply_visitor(cluster_subblock_visitor(cluster_subblock_visitor::BLOCK_OUTPUT), subblock);
}

void
cluster_subblock_output_set(sprokit::cluster_subblock_t& subblock, sprokit::cluster_output_t const& output)
{
  subblock = output;
}

object
cluster_block_config(sprokit::cluster_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_CONFIG), block);
}

void
cluster_block_config_set(sprokit::cluster_block& block, sprokit::config_pipe_block const& config)
{
  block = config;
}

object
cluster_block_process(sprokit::cluster_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_PROCESS), block);
}

void
cluster_block_process_set(sprokit::cluster_block& block, sprokit::process_pipe_block const& process)
{
  block = process;
}

object
cluster_block_connect(sprokit::cluster_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_CONNECT), block);
}

void
cluster_block_connect_set(sprokit::cluster_block& block, sprokit::connect_pipe_block const& connect)
{
  block = connect;
}

object
cluster_block_cluster(sprokit::cluster_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_CLUSTER), block);
}

void
cluster_block_cluster_set(sprokit::cluster_block& block, sprokit::cluster_pipe_block const& cluster)
{
  block = cluster;
}

sprokit::pipe_blocks
load_pipe_file(std::string const& path)
{
  return sprokit::load_pipe_blocks_from_file(path);
}

sprokit::pipe_blocks
load_pipe(object const& stream)
{
  sprokit::python::pyistream istr(stream);

  return sprokit::load_pipe_blocks(istr);
}

sprokit::cluster_blocks
load_cluster_file(std::string const& path)
{
  return sprokit::load_cluster_blocks_from_file(path);
}

sprokit::cluster_blocks
load_cluster(object const& stream)
{
  sprokit::python::pyistream istr(stream);

  return sprokit::load_cluster_blocks(istr);
}

pipe_block_visitor
::pipe_block_visitor(block_t type)
  : block_type(type)
{
}

pipe_block_visitor
::~pipe_block_visitor()
{
}

object
pipe_block_visitor
::operator () (sprokit::config_pipe_block const& config_block) const
{
  sprokit::python::python_gil const gil;

  (void)gil;

  object obj;

  if (block_type == BLOCK_CONFIG)
  {
    obj = object(config_block);
  }

  return obj;
}

object
pipe_block_visitor
::operator () (sprokit::process_pipe_block const& process_block) const
{
  sprokit::python::python_gil const gil;

  (void)gil;

  object obj;

  if (block_type == BLOCK_PROCESS)
  {
    obj = object(process_block);
  }

  return obj;
}

object
pipe_block_visitor
::operator () (sprokit::connect_pipe_block const& connect_block) const
{
  sprokit::python::python_gil const gil;

  (void)gil;

  object obj;

  if (block_type == BLOCK_CONNECT)
  {
    obj = object(connect_block);
  }

  return obj;
}

object
pipe_block_visitor
::operator () (sprokit::cluster_pipe_block const& cluster_block) const
{
  sprokit::python::python_gil const gil;

  (void)gil;

  object obj;

  if (block_type == BLOCK_CLUSTER)
  {
    obj = object(cluster_block);
  }

  return obj;
}

cluster_subblock_visitor
::cluster_subblock_visitor(block_t type)
  : block_type(type)
{
}

cluster_subblock_visitor
::~cluster_subblock_visitor()
{
}

object
cluster_subblock_visitor
::operator () (sprokit::cluster_config_t const& config) const
{
  sprokit::python::python_gil const gil;

  (void)gil;

  if (block_type == BLOCK_CONFIG)
  {
    return object(config);
  }

  return object();
}

object
cluster_subblock_visitor
::operator () (sprokit::cluster_input_t const& input) const
{
  sprokit::python::python_gil const gil;

  (void)gil;

  if (block_type == BLOCK_INPUT)
  {
    return object(input);
  }

  return object();
}

object
cluster_subblock_visitor
::operator () (sprokit::cluster_output_t const& output) const
{
  sprokit::python::python_gil const gil;

  (void)gil;

  if (block_type == BLOCK_OUTPUT)
  {
    return object(output);
  }

  return object();
}
