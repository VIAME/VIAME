/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/pystream.h>

#include <vistk/pipeline_util/load_pipe.h>
#include <vistk/pipeline_util/load_pipe_exception.h>
#include <vistk/pipeline_util/pipe_declaration_types.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <string>

/**
 * \file load.cxx
 *
 * \brief Python bindings for loading pipe blocks.
 */

using namespace boost::python;

static object config_key_options_flags(vistk::config_key_options_t const& options);
static void config_key_options_flags_set(vistk::config_key_options_t& options, vistk::config_flags_t const& flags);
static object config_key_options_provider(vistk::config_key_options_t const& options);
static void config_key_options_provider_set(vistk::config_key_options_t& options, vistk::config_provider_t const& provider);
static object map_options_flags(vistk::map_options_t const& options);
static void map_options_flags_set(vistk::map_options_t& options, vistk::process::port_flags_t const& flags);
static object pipe_block_config(vistk::pipe_block const& block);
static void pipe_block_config_set(vistk::pipe_block& block, vistk::config_pipe_block const& config);
static object pipe_block_process(vistk::pipe_block const& block);
static void pipe_block_process_set(vistk::pipe_block& block, vistk::process_pipe_block const& process);
static object pipe_block_connect(vistk::pipe_block const& block);
static void pipe_block_connect_set(vistk::pipe_block& block, vistk::connect_pipe_block const& connect);
static object pipe_block_group(vistk::pipe_block const& block);
static void pipe_block_group_set(vistk::pipe_block& block, vistk::group_pipe_block const& group);
static vistk::pipe_blocks load_pipe_file(std::string const& path);
static vistk::pipe_blocks load_pipe(object const& stream, std::string const& inc_root = "");
static void translator(vistk::load_pipe_exception const& e);

BOOST_PYTHON_FUNCTION_OVERLOADS(load_pipe_overloads, load_pipe, 1, 2);

BOOST_PYTHON_MODULE(load)
{
  register_exception_translator<
    vistk::load_pipe_exception>(translator);

  class_<vistk::token_t>("Token");
  class_<vistk::config_flag_t>("ConfigFlag");
  class_<vistk::config_flags_t>("ConfigFlags")
    .def(vector_indexing_suite<vistk::config_flags_t>())
  ;
  class_<vistk::config_provider_t>("ConfigProvider");
  class_<vistk::config_key_options_t>("ConfigKeyOptions")
    .add_property("flags", &config_key_options_flags, &config_key_options_flags_set)
    .add_property("provider", &config_key_options_provider, &config_key_options_provider_set)
  ;
  class_<vistk::config_key_t>("ConfigKey")
    .def_readwrite("key_path", &vistk::config_key_t::key_path)
    .def_readwrite("options", &vistk::config_key_t::options)
  ;
  class_<vistk::config_value_t>("ConfigValue")
    .def_readwrite("key", &vistk::config_value_t::key)
    .def_readwrite("value", &vistk::config_value_t::value)
  ;
  class_<vistk::config_values_t>("ConfigValues")
    /// \todo Need operator == on config_value_t
    //.def(vector_indexing_suite<vistk::config_values_t>())
  ;
  class_<vistk::map_options_t>("MapOptions")
    .add_property("flags", &map_options_flags, &map_options_flags_set)
  ;
  class_<vistk::input_map_t>("InputMap")
    .def_readwrite("options", &vistk::input_map_t::options)
    .def_readwrite("from", &vistk::input_map_t::from)
    .def_readwrite("to", &vistk::input_map_t::to)
  ;
  class_<vistk::input_maps_t>("InputMaps")
    /// \todo Need operator == on input_map_t.
    //.def(vector_indexing_suite<vistk::input_maps_t>())
  ;
  class_<vistk::output_map_t>("OutputMap")
    .def_readwrite("options", &vistk::output_map_t::options)
    .def_readwrite("from", &vistk::output_map_t::from)
    .def_readwrite("to", &vistk::output_map_t::to)
  ;
  class_<vistk::output_maps_t>("OutputMaps")
    /// \todo Need operator == on output_map_t.
    //.def(vector_indexing_suite<vistk::output_maps_t>())
  ;
  class_<vistk::config_pipe_block>("ConfigBlock")
    .def_readwrite("key", &vistk::config_pipe_block::key)
    .def_readwrite("values", &vistk::config_pipe_block::values)
  ;
  class_<vistk::process_pipe_block>("ProcessBlock")
    .def_readwrite("name", &vistk::process_pipe_block::name)
    .def_readwrite("type", &vistk::process_pipe_block::type)
    .def_readwrite("config_values", &vistk::process_pipe_block::config_values)
  ;
  class_<vistk::connect_pipe_block>("ConnectBlock")
    .def_readwrite("from", &vistk::connect_pipe_block::from)
    .def_readwrite("to", &vistk::connect_pipe_block::to)
  ;
  class_<vistk::group_pipe_block>("GroupBlock")
    .def_readwrite("name", &vistk::group_pipe_block::name)
    .def_readwrite("config_values", &vistk::group_pipe_block::config_values)
    .def_readwrite("input_mappings", &vistk::group_pipe_block::input_mappings)
    .def_readwrite("output_mappings", &vistk::group_pipe_block::output_mappings)
  ;
  class_<vistk::pipe_block>("PipeBlock")
    .add_property("config", &pipe_block_config, &pipe_block_config_set)
    .add_property("process", &pipe_block_process, &pipe_block_process_set)
    .add_property("connect", &pipe_block_connect, &pipe_block_connect_set)
    .add_property("group", &pipe_block_group, &pipe_block_group_set)
  ;
  class_<vistk::pipe_blocks>("PipeBlocks")
    /// \todo Need operator == on pipe_block.
    //.def(vector_indexing_suite<vistk::pipe_blocks>())
  ;

  def("load_pipe_file", &load_pipe_file);
  def("load_pipe", &load_pipe, load_pipe_overloads());
}

void
translator(vistk::load_pipe_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

object
config_key_options_flags(vistk::config_key_options_t const& options)
{
  if (options.flags)
  {
    return object(*options.flags);
  }

  return object();
}

void
config_key_options_flags_set(vistk::config_key_options_t& options, vistk::config_flags_t const& flags)
{
  options.flags = flags;
}

object
config_key_options_provider(vistk::config_key_options_t const& options)
{
  if (options.provider)
  {
    return object(*options.provider);
  }

  return object();
}

void
config_key_options_provider_set(vistk::config_key_options_t& options, vistk::config_provider_t const& provider)
{
  options.provider = provider;
}

object
map_options_flags(vistk::map_options_t const& options)
{
  if (options.flags)
  {
    return object(*options.flags);
  }

  return object();
}

void
map_options_flags_set(vistk::map_options_t& options, vistk::process::port_flags_t const& flags)
{
  options.flags = flags;
}

object
pipe_block_config(vistk::pipe_block const& block)
{
  /// \todo Get a config block from the block.

  return object();
}

void
pipe_block_config_set(vistk::pipe_block& block, vistk::config_pipe_block const& config)
{
  block = config;
}

object
pipe_block_process(vistk::pipe_block const& block)
{
  /// \todo Get a process block from the block.

  return object();
}

void
pipe_block_process_set(vistk::pipe_block& block, vistk::process_pipe_block const& process)
{
  block = process;
}

object
pipe_block_connect(vistk::pipe_block const& block)
{
  /// \todo Get a connect block from the block.

  return object();
}

void
pipe_block_connect_set(vistk::pipe_block& block, vistk::connect_pipe_block const& connect)
{
  block = connect;
}

object
pipe_block_group(vistk::pipe_block const& block)
{
  /// \todo Get a group block from the block.

  return object();
}

void
pipe_block_group_set(vistk::pipe_block& block, vistk::group_pipe_block const& group)
{
  block = group;
}

vistk::pipe_blocks
load_pipe_file(std::string const& path)
{
  return vistk::load_pipe_blocks_from_file(boost::filesystem::path(path));
}

vistk::pipe_blocks
load_pipe(object const& stream, std::string const& inc_root)
{
  pyistream istr(stream);

  return vistk::load_pipe_blocks(istr, boost::filesystem::path(inc_root));
}
