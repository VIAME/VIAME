/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/utils.h>

extern "C"
{
#include <lua.h>
}

#include <luabind/luabind.hpp>
#include <luabind/class.hpp>
#include <luabind/function.hpp>

/**
 * \file utils.cxx
 *
 * \brief Lua bindings for utils.
 */

extern "C"
{

int luaopen_vistk_pipeline_utils(lua_State* L);

}

using namespace luabind;

int
luaopen_vistk_pipeline_utils(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline")
    [
      class_<vistk::thread_name_t>("thread_name")
        .def(constructor<>())
    , def("name_thread", &vistk::name_thread)
    ]
  ];

  return 0;
}
