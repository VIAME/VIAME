/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <lua/helpers/lua_convert_optional.h>
#include <lua/helpers/lua_convert_vector.h>
#include <lua/helpers/luastream.h>

#include <vistk/pipeline_util/load_pipe.h>
#include <vistk/pipeline_util/pipe_declaration_types.h>

extern "C"
{
#include <lua.h>
}

#include <luabind/class.hpp>
#include <luabind/function.hpp>

#include <string>

/**
 * \file load.cxx
 *
 * \brief Lua bindings for loading pipe blocks.
 */

using namespace luabind;

extern "C"
{

int luaopen_vistk_pipeline_util_load(lua_State* L);

}

static void pipe_block_config(lua_State* L, vistk::pipe_block const& block);
static void pipe_block_config_set(vistk::pipe_block& block, vistk::config_pipe_block const& config);
static void pipe_block_process(lua_State* L, vistk::pipe_block const& block);
static void pipe_block_process_set(vistk::pipe_block& block, vistk::process_pipe_block const& process);
static void pipe_block_connect(lua_State* L, vistk::pipe_block const& block);
static void pipe_block_connect_set(vistk::pipe_block& block, vistk::connect_pipe_block const& connect);
static void pipe_block_group(lua_State* L, vistk::pipe_block const& block);
static void pipe_block_group_set(vistk::pipe_block& block, vistk::group_pipe_block const& group);
static vistk::pipe_blocks load_pipe_file(std::string const& path);
static vistk::pipe_blocks load_pipe(object const& stream, std::string const& inc_root);
static vistk::pipe_blocks default_load_pipe(object const& stream);

class block_visitor
  : public boost::static_visitor<>
{
  public:
    typedef enum
    {
      BLOCK_CONFIG,
      BLOCK_PROCESS,
      BLOCK_CONNECT,
      BLOCK_GROUP
    } block_t;

    block_visitor(block_t type, lua_State* L);
    ~block_visitor();

    block_t const block_type;
    lua_State* interpreter;

    void operator () (vistk::config_pipe_block const& config_block) const;
    void operator () (vistk::process_pipe_block const& process_block) const;
    void operator () (vistk::connect_pipe_block const& connect_block) const;
    void operator () (vistk::group_pipe_block const& group_block) const;
};

int
luaopen_vistk_pipeline_util_load(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline_util")
    [
      class_<vistk::token_t>("token")
        .def(constructor<>())
    , class_<vistk::config_flag_t>("config_flag")
        .def(constructor<>())
    , class_<vistk::config_flags_t>("config_flags")
        .def(constructor<>())
    , class_<vistk::config_provider_t>("config_provider")
        .def(constructor<>())
    , class_<vistk::config_key_options_t>("config_key_options")
        .def(constructor<>())
        .def_readwrite("flags", &vistk::config_key_options_t::flags)
        .def_readwrite("provider", &vistk::config_key_options_t::provider)
    , class_<vistk::config_key_t>("config_key")
        .def(constructor<>())
        .def_readwrite("key_path", &vistk::config_key_t::key_path)
        .def_readwrite("options", &vistk::config_key_t::options)
    , class_<vistk::config_value_t>("config_value")
        .def(constructor<>())
        .def_readwrite("key", &vistk::config_value_t::key)
        .def_readwrite("value", &vistk::config_value_t::value)
    , class_<vistk::config_values_t>("config_values")
        .def(constructor<>())
        /// \todo Need operator == on config_value_t
    , class_<vistk::map_options_t>("map_options")
        .def(constructor<>())
        .def_readwrite("flags", &vistk::map_options_t::flags)
    , class_<vistk::input_map_t>("input_map")
        .def(constructor<>())
        .def_readwrite("options", &vistk::input_map_t::options)
        .def_readwrite("from", &vistk::input_map_t::from)
        .def_readwrite("to", &vistk::input_map_t::to)
    , class_<vistk::input_maps_t>("input_maps")
        .def(constructor<>())
        /// \todo Need operator == on input_map_t.
    , class_<vistk::output_map_t>("output_map")
        .def(constructor<>())
        .def_readwrite("options", &vistk::output_map_t::options)
        .def_readwrite("from", &vistk::output_map_t::from)
        .def_readwrite("to", &vistk::output_map_t::to)
    , class_<vistk::output_maps_t>("output_maps")
        .def(constructor<>())
        /// \todo Need operator == on output_map_t.
    , class_<vistk::config_pipe_block>("config_block")
        .def(constructor<>())
        .def_readwrite("key", &vistk::config_pipe_block::key)
        .def_readwrite("values", &vistk::config_pipe_block::values)
    , class_<vistk::process_pipe_block>("process_block")
        .def(constructor<>())
        .def_readwrite("name", &vistk::process_pipe_block::name)
        .def_readwrite("type", &vistk::process_pipe_block::type)
        .def_readwrite("config_values", &vistk::process_pipe_block::config_values)
    , class_<vistk::connect_pipe_block>("connect_block")
        .def(constructor<>())
        .def_readwrite("from", &vistk::connect_pipe_block::from)
        .def_readwrite("to", &vistk::connect_pipe_block::to)
    , class_<vistk::group_pipe_block>("group_block")
        .def(constructor<>())
        .def_readwrite("name", &vistk::group_pipe_block::name)
        .def_readwrite("config_values", &vistk::group_pipe_block::config_values)
        .def_readwrite("input_mappings", &vistk::group_pipe_block::input_mappings)
        .def_readwrite("output_mappings", &vistk::group_pipe_block::output_mappings)
    , class_<vistk::pipe_block>("pipe_block")
        .def(constructor<>())
        .property("config", &pipe_block_config, &pipe_block_config_set)
        .property("process", &pipe_block_process, &pipe_block_process_set)
        .property("connect", &pipe_block_connect, &pipe_block_connect_set)
        .property("group", &pipe_block_group, &pipe_block_group_set)
    , class_<vistk::pipe_blocks>("pipe_blocks")
        .def(constructor<>())
        /// \todo Need operator == on pipe_block.
    , def("load_pipe_file", &load_pipe_file)
    , def("load_pipe", &load_pipe)
    , def("load_pipe", &default_load_pipe)
    ]
  ];

  return 0;
}

void
pipe_block_config(lua_State* L, vistk::pipe_block const& block)
{
  boost::apply_visitor(block_visitor(block_visitor::BLOCK_CONFIG, L), block);
}

void
pipe_block_config_set(vistk::pipe_block& block, vistk::config_pipe_block const& config)
{
  block = config;
}

void
pipe_block_process(lua_State* L, vistk::pipe_block const& block)
{
  boost::apply_visitor(block_visitor(block_visitor::BLOCK_PROCESS, L), block);
}

void
pipe_block_process_set(vistk::pipe_block& block, vistk::process_pipe_block const& process)
{
  block = process;
}

void
pipe_block_connect(lua_State* L, vistk::pipe_block const& block)
{
  boost::apply_visitor(block_visitor(block_visitor::BLOCK_CONNECT, L), block);
}

void
pipe_block_connect_set(vistk::pipe_block& block, vistk::connect_pipe_block const& connect)
{
  block = connect;
}

void
pipe_block_group(lua_State* L, vistk::pipe_block const& block)
{
  boost::apply_visitor(block_visitor(block_visitor::BLOCK_GROUP, L), block);
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
  luaistream istr(stream);

  return vistk::load_pipe_blocks(istr, boost::filesystem::path(inc_root));
}

vistk::pipe_blocks
default_load_pipe(object const& stream)
{
  luaistream istr(stream);

  return vistk::load_pipe_blocks(istr);
}

block_visitor
::block_visitor(block_t type, lua_State* L)
  : block_type(type)
  , interpreter(L)
{
}

block_visitor
::~block_visitor()
{
}

void
block_visitor
::operator () (vistk::config_pipe_block const& config_block) const
{
  if (block_type == BLOCK_CONFIG)
  {
    detail::convert_to_lua(interpreter, config_block);
    return;
  }

  lua_pushnil(interpreter);
}

void
block_visitor
::operator () (vistk::process_pipe_block const& process_block) const
{
  if (block_type == BLOCK_PROCESS)
  {
    detail::convert_to_lua(interpreter, process_block);
    return;
  }

  lua_pushnil(interpreter);
}

void
block_visitor
::operator () (vistk::connect_pipe_block const& connect_block) const
{
  if (block_type == BLOCK_CONNECT)
  {
    detail::convert_to_lua(interpreter, connect_block);
    return;
  }

  lua_pushnil(interpreter);
}

void
block_visitor
::operator () (vistk::group_pipe_block const& group_block) const
{
  if (block_type == BLOCK_GROUP)
  {
    detail::convert_to_lua(interpreter, group_block);
    return;
  }

  lua_pushnil(interpreter);
}
