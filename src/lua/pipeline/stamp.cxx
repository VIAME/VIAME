/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/stamp.h>

extern "C"
{
#include <lua.h>
}

#include <luabind/luabind.hpp>
#include <luabind/class.hpp>
#include <luabind/function.hpp>
#include <luabind/operator.hpp>

/**
 * \file stamp.cxx
 *
 * \brief Lua bindings for \link vistk::stamp\endlink.
 */

extern "C"
{

int luaopen_vistk_pipeline_stamp(lua_State* L);

}

using namespace luabind;

int
luaopen_vistk_pipeline_stamp(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline")
    [
      def("new_stamp", &vistk::stamp::new_stamp)
    , def("copied_stamp", &vistk::stamp::copied_stamp)
    , def("incremented_stamp", &vistk::stamp::incremented_stamp)
    , def("recolored_stamp", &vistk::stamp::recolored_stamp)
    , class_<vistk::stamp, vistk::stamp_t>("stamp")
        .def("is_same_color", &vistk::stamp::is_same_color)
        .def(self == self)
        .def(self < self)
    ]
  ];

  return 0;
}
