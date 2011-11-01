/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/pipeline.h>

#include <vistk/pipeline_util/pipe_bakery.h>

#include <lua/helpers/lua_include.h>
#include <lua/helpers/luastream.h>

#include <luabind/class.hpp>
#include <luabind/function.hpp>

#include <string>

/**
 * \file bake.cxx
 *
 * \brief Lua bindings for baking pipelines.
 */

using namespace luabind;

extern "C"
{

int luaopen_vistk_pipeline_util_bake(lua_State* L);

}

static vistk::pipeline_t bake_pipe_file(std::string const& path);
static vistk::pipeline_t bake_pipe(object stream, std::string const& inc_root);
static vistk::pipeline_t default_bake_pipe(object stream);

int
luaopen_vistk_pipeline_util_bake(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline_util")
    [
      def("bake_pipe_file", &bake_pipe_file)
    , def("bake_pipe", &bake_pipe)
    , def("bake_pipe", &default_bake_pipe)
    , def("bake_pipe_blocks", &vistk::bake_pipe_blocks)
    , def("extract_configuration", &vistk::extract_configuration)
    ]
  ];

  return 0;
}

vistk::pipeline_t
bake_pipe_file(std::string const& path)
{
  return vistk::bake_pipe_from_file(boost::filesystem::path(path));
}

vistk::pipeline_t
bake_pipe(object stream, std::string const& inc_root)
{
  luaistream istr(stream);

  return vistk::bake_pipe(istr, boost::filesystem::path(inc_root));
}

vistk::pipeline_t
default_bake_pipe(object stream)
{
  luaistream istr(stream);

  return vistk::bake_pipe(istr);
}
