/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <lua/helpers/lua_convert_vector.h>

#include <vistk/pipeline/config.h>

extern "C"
{
#include <lua.h>
}

#include <luabind/class.hpp>
#include <luabind/function.hpp>
#include <luabind/iterator_policy.hpp>

/**
 * \file config.cxx
 *
 * \brief Lua bindings for \link vistk::config\endlink.
 */

using namespace luabind;

extern "C"
{

int luaopen_vistk_pipeline_config(lua_State* L);

}

static vistk::config_t default_empty_config();
static vistk::config::value_t config_get_value(vistk::config_t conf, vistk::config::key_t const& key);
static vistk::config::value_t config_get_value_with_default(vistk::config_t conf, vistk::config::key_t const& key, vistk::config::value_t const& def);

int
luaopen_vistk_pipeline_config(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline")
    [
      def("empty_config", &vistk::config::empty_config)
    , def("empty_config", &default_empty_config)
    , class_<vistk::config::key_t>("config_key")
        .def(constructor<>())
    , class_<vistk::config::keys_t>("config_keys")
        .def(constructor<>())
    , class_<vistk::config::value_t>("config_value")
        .def(constructor<>())
    , class_<vistk::config, vistk::config_t>("config")
        .def("subblock", &vistk::config::subblock)
        .def("subblock_view", &vistk::config::subblock_view
          , dependency(result, _1))
        .def("get_value", &config_get_value)
        .def("get_value", &config_get_value_with_default)
        .def("set_value", &vistk::config::set_value)
        .def("unset_value", &vistk::config::unset_value)
        .def("is_read_only", &vistk::config::is_read_only)
        .def("mark_read_only", &vistk::config::mark_read_only)
        .def("merge_config", &vistk::config::merge_config)
        .def("available_values", &vistk::config::available_values
          , return_stl_iterator)
        .def("has_value", &vistk::config::has_value)
        //.scope
        //[
        //  def("block_sep", &vistk::config::block_sep)
        //, def("global_value", &vistk::config::global_value)
        //]
    ]
  ];

  return 0;
}

vistk::config_t
default_empty_config()
{
  return vistk::config::empty_config();
}

vistk::config::value_t
config_get_value(vistk::config_t conf, vistk::config::key_t const& key)
{
  return conf->get_value<vistk::config::value_t>(key);
}

vistk::config::value_t
config_get_value_with_default(vistk::config_t conf, vistk::config::key_t const& key, vistk::config::value_t const& def)
{
  return conf->get_value<vistk::config::value_t>(key, def);
}
