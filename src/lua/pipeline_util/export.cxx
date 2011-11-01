/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline_util/export_dot.h>

#include <lua/helpers/lua_include.h>
#include <lua/helpers/luastream.h>

#include <luabind/function.hpp>

#include <string>

/**
 * \file export.cxx
 *
 * \brief Lua bindings for exporting functions.
 */

using namespace luabind;

extern "C"
{

int luaopen_vistk_pipeline_util_export(lua_State* L);

}

void export_dot(object const& stream, vistk::pipeline_t const pipe, std::string const& graph_name);

int
luaopen_vistk_pipeline_util_export(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline_util")
    [
      def("export_dot", &export_dot)
    ]
  ];

  return 0;
}

void
export_dot(object const& stream, vistk::pipeline_t const pipe, std::string const& graph_name)
{
  luaostream ostr(stream);

  return vistk::export_dot(ostr, pipe, graph_name);
}
