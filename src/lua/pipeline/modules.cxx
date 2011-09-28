/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/modules.h>

extern "C"
{
#include <lua.h>
}

#include <luabind/luabind.hpp>
#include <luabind/function.hpp>

/**
 * \file modules.cxx
 *
 * \brief Lua bindings for module loading.
 */

extern "C"
{

int luaopen_vistk_pipeline_modules(lua_State* L);

}

using namespace luabind;

int
luaopen_vistk_pipeline_modules(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline")
    [
      def("load_known_modules", &vistk::load_known_modules)
    ]
  ];

  return 0;
}
