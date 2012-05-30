/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/pystream.h>
#include <python/helpers/python_convert_optional.h>

#include <vistk/pipeline_util/load_pipe.h>
#include <vistk/pipeline_util/load_pipe_exception.h>
#include <vistk/pipeline_util/pipe_declaration_types.h>

#include <vistk/pipeline/process.h>

#include <vistk/utilities/path.h>

#include <vistk/python/util/python_gil.h>

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

static object group_subblock_config(vistk::group_subblock_t const& subblock);
static void group_subblock_config_set(vistk::group_subblock_t& subblock, vistk::config_value_t const& config);
static object group_subblock_input(vistk::group_subblock_t const& subblock);
static void group_subblock_input_set(vistk::group_subblock_t& subblock, vistk::input_map_t const& input);
static object group_subblock_output(vistk::group_subblock_t const& subblock);
static void group_subblock_output_set(vistk::group_subblock_t& subblock, vistk::output_map_t const& output);
static object pipe_block_config(vistk::pipe_block const& block);
static void pipe_block_config_set(vistk::pipe_block& block, vistk::config_pipe_block const& config);
static object pipe_block_process(vistk::pipe_block const& block);
static void pipe_block_process_set(vistk::pipe_block& block, vistk::process_pipe_block const& process);
static object pipe_block_connect(vistk::pipe_block const& block);
static void pipe_block_connect_set(vistk::pipe_block& block, vistk::connect_pipe_block const& connect);
static object pipe_block_group(vistk::pipe_block const& block);
static void pipe_block_group_set(vistk::pipe_block& block, vistk::group_pipe_block const& group);
static vistk::pipe_blocks load_pipe_file(std::string const& path);
static vistk::pipe_blocks load_pipe(object const& stream, std::string const& inc_root);

BOOST_PYTHON_MODULE(load)
{
  register_optional_converter<vistk::config_flags_t>("ConfigFlagsOpt", "An optional config flags.");
  register_optional_converter<vistk::config_provider_t>("ConfigProviderOpt", "An optional config provider.");
  register_optional_converter<vistk::process::port_flags_t>("PortFlagsOpt", "An optional port flags.");

  class_<vistk::token_t>("Token"
    , "A token in the pipeline description.");
  class_<vistk::config_flag_t>("ConfigFlag"
    , "A flag on a configuration setting.");
  class_<vistk::config_flags_t>("ConfigFlags"
    , "A collection of flags on a configuration setting.")
    .def(vector_indexing_suite<vistk::config_flags_t>())
  ;
  class_<vistk::config_provider_t>("ConfigProvider"
    , "A provider key for a configuration setting.");
  class_<vistk::config_key_options_t>("ConfigKeyOptions"
    , "A collection of options on a configuration setting.")
    .def_readwrite("flags", &vistk::config_key_options_t::flags)
    .def_readwrite("provider", &vistk::config_key_options_t::provider)
  ;
  class_<vistk::config_key_t>("ConfigKey"
    , "A configuration key with its settings.")
    .def_readwrite("key_path", &vistk::config_key_t::key_path)
    .def_readwrite("options", &vistk::config_key_t::options)
  ;
  class_<vistk::config_value_t>("ConfigValue"
    , "A complete configuration setting.")
    .def_readwrite("key", &vistk::config_value_t::key)
    .def_readwrite("value", &vistk::config_value_t::value)
  ;
  class_<vistk::config_values_t>("ConfigValues"
    , "A collection of configuration settings.")
    /// \todo Need operator == on config_value_t
    //.def(vector_indexing_suite<vistk::config_values_t>())
  ;
  class_<vistk::map_options_t>("MapOptions"
    , "A collection of options for a mapping.")
    .def_readwrite("flags", &vistk::map_options_t::flags)
  ;
  class_<vistk::input_map_t>("InputMap"
    , "An input mapping for a group.")
    .def_readwrite("options", &vistk::input_map_t::options)
    .def_readwrite("from_", &vistk::input_map_t::from)
    .def_readwrite("to", &vistk::input_map_t::to)
  ;
  class_<vistk::output_map_t>("OutputMap"
    , "An output mapping for a group.")
    .def_readwrite("options", &vistk::output_map_t::options)
    .def_readwrite("from_", &vistk::output_map_t::from)
    .def_readwrite("to", &vistk::output_map_t::to)
  ;
  class_<vistk::config_pipe_block>("ConfigBlock"
    , "A block of configuration settings.")
    .def_readwrite("key", &vistk::config_pipe_block::key)
    .def_readwrite("values", &vistk::config_pipe_block::values)
  ;
  class_<vistk::process_pipe_block>("ProcessBlock"
    , "A block which declares a process.")
    .def_readwrite("name", &vistk::process_pipe_block::name)
    .def_readwrite("type", &vistk::process_pipe_block::type)
    .def_readwrite("config_values", &vistk::process_pipe_block::config_values)
  ;
  class_<vistk::connect_pipe_block>("ConnectBlock"
    , "A block which connects two ports together.")
    .def_readwrite("from_", &vistk::connect_pipe_block::from)
    .def_readwrite("to", &vistk::connect_pipe_block::to)
  ;
  class_<vistk::group_subblock_t>("GroupSubblock"
    , "A subblock within a group.")
    .add_property("config", &group_subblock_config, &group_subblock_config_set)
    .add_property("input", &group_subblock_input, &group_subblock_input_set)
    .add_property("output", &group_subblock_output, &group_subblock_output_set)
  ;
  class_<vistk::group_subblocks_t>("GroupSubblocks"
    , "A collection of group subblocks.")
    /// \todo Need operator == on group_subblock_t.
    //.def(vector_indexing_suite<vistk::group_subblocks_t>())
  ;
  class_<vistk::group_pipe_block>("GroupBlock"
    , "A block which declares a group within the pipeline.")
    .def_readwrite("name", &vistk::group_pipe_block::name)
    .def_readwrite("subblocks", &vistk::group_pipe_block::subblocks)
  ;
  class_<vistk::pipe_block>("PipeBlock"
    , "A block in a pipeline declaration file.")
    .add_property("config", &pipe_block_config, &pipe_block_config_set)
    .add_property("process", &pipe_block_process, &pipe_block_process_set)
    .add_property("connect", &pipe_block_connect, &pipe_block_connect_set)
    .add_property("group", &pipe_block_group, &pipe_block_group_set)
  ;
  class_<vistk::pipe_blocks>("PipeBlocks"
    , "A collection of pipeline blocks.")
    /// \todo Need operator == on pipe_block.
    //.def(vector_indexing_suite<vistk::pipe_blocks>())
  ;

  def("load_pipe_file", &load_pipe_file
    , (arg("path"))
    , "Load pipe blocks from a file.");
  def("load_pipe", &load_pipe
    , (arg("stream"), arg("inc_root") = std::string())
    , "Load pipe blocks from a stream.");
}

class group_subblock_visitor
  : public boost::static_visitor<object>
{
  public:
    typedef enum
    {
      BLOCK_CONFIG,
      BLOCK_INPUT,
      BLOCK_OUTPUT
    } block_t;

    group_subblock_visitor(block_t type);
    ~group_subblock_visitor();

    block_t const block_type;

    object operator () (vistk::config_value_t const& config) const;
    object operator () (vistk::input_map_t const& input) const;
    object operator () (vistk::output_map_t const& output) const;
};

object
group_subblock_config(vistk::group_subblock_t const& subblock)
{
  return boost::apply_visitor(group_subblock_visitor(group_subblock_visitor::BLOCK_CONFIG), subblock);
}

void
group_subblock_config_set(vistk::group_subblock_t& subblock, vistk::config_value_t const& config)
{
  subblock = config;
}

object
group_subblock_input(vistk::group_subblock_t const& subblock)
{
  return boost::apply_visitor(group_subblock_visitor(group_subblock_visitor::BLOCK_INPUT), subblock);
}

void
group_subblock_input_set(vistk::group_subblock_t& subblock, vistk::input_map_t const& input)
{
  subblock = input;
}

object
group_subblock_output(vistk::group_subblock_t const& subblock)
{
  return boost::apply_visitor(group_subblock_visitor(group_subblock_visitor::BLOCK_OUTPUT), subblock);
}

void
group_subblock_output_set(vistk::group_subblock_t& subblock, vistk::output_map_t const& output)
{
  subblock = output;
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
      BLOCK_GROUP
    } block_t;

    pipe_block_visitor(block_t type);
    ~pipe_block_visitor();

    block_t const block_type;

    object operator () (vistk::config_pipe_block const& config_block) const;
    object operator () (vistk::process_pipe_block const& process_block) const;
    object operator () (vistk::connect_pipe_block const& connect_block) const;
    object operator () (vistk::group_pipe_block const& group_block) const;
};

object
pipe_block_config(vistk::pipe_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_CONFIG), block);
}

void
pipe_block_config_set(vistk::pipe_block& block, vistk::config_pipe_block const& config)
{
  block = config;
}

object
pipe_block_process(vistk::pipe_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_PROCESS), block);
}

void
pipe_block_process_set(vistk::pipe_block& block, vistk::process_pipe_block const& process)
{
  block = process;
}

object
pipe_block_connect(vistk::pipe_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_CONNECT), block);
}

void
pipe_block_connect_set(vistk::pipe_block& block, vistk::connect_pipe_block const& connect)
{
  block = connect;
}

object
pipe_block_group(vistk::pipe_block const& block)
{
  return boost::apply_visitor(pipe_block_visitor(pipe_block_visitor::BLOCK_GROUP), block);
}

void
pipe_block_group_set(vistk::pipe_block& block, vistk::group_pipe_block const& group)
{
  block = group;
}

vistk::pipe_blocks
load_pipe_file(std::string const& path)
{
  return vistk::load_pipe_blocks_from_file(vistk::path_t(path));
}

vistk::pipe_blocks
load_pipe(object const& stream, std::string const& inc_root)
{
  pyistream istr(stream);

  return vistk::load_pipe_blocks(istr, vistk::path_t(inc_root));
}

group_subblock_visitor
::group_subblock_visitor(block_t type)
  : block_type(type)
{
}

group_subblock_visitor
::~group_subblock_visitor()
{
}

object
group_subblock_visitor
::operator () (vistk::config_value_t const& config) const
{
  vistk::python::python_gil const gil;

  (void)gil;

  if (block_type == BLOCK_CONFIG)
  {
    return object(config);
  }

  return object();
}

object
group_subblock_visitor
::operator () (vistk::input_map_t const& input) const
{
  vistk::python::python_gil const gil;

  (void)gil;

  if (block_type == BLOCK_INPUT)
  {
    return object(input);
  }

  return object();
}

object
group_subblock_visitor
::operator () (vistk::output_map_t const& output) const
{
  vistk::python::python_gil const gil;

  (void)gil;

  if (block_type == BLOCK_OUTPUT)
  {
    return object(output);
  }

  return object();
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
::operator () (vistk::config_pipe_block const& config_block) const
{
  vistk::python::python_gil const gil;

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
::operator () (vistk::process_pipe_block const& process_block) const
{
  vistk::python::python_gil const gil;

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
::operator () (vistk::connect_pipe_block const& connect_block) const
{
  vistk::python::python_gil const gil;

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
::operator () (vistk::group_pipe_block const& group_block) const
{
  vistk::python::python_gil const gil;

  (void)gil;

  object obj;

  if (block_type == BLOCK_GROUP)
  {
    obj = object(group_block);
  }

  return obj;
}
