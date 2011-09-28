/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/datum.h>

extern "C"
{
#include <lua.h>
}

#include <luabind/luabind.hpp>
#include <luabind/class.hpp>
#include <luabind/function.hpp>

/**
 * \file datum.cxx
 *
 * \brief Lua bindings for \link vistk::datum\endlink.
 */

extern "C"
{

int luaopen_vistk_pipeline_datum(lua_State* L);

}

using namespace luabind;

int
luaopen_vistk_pipeline_datum(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline")
    [
      class_<vistk::datum::error_t>("datum_error")
    /// \todo How to create a new datum packet?
    //, def("new_datum", &vistk::datum::new_datum)
    , def("empty_datum", &vistk::datum::empty_datum)
    , def("complete_datum", &vistk::datum::complete_datum)
    , def("error_datum", &vistk::datum::error_datum)
    , class_<vistk::datum, vistk::datum_t>("datum")
      .enum_("types")
        [
          value("invalid", vistk::datum::invalid)
        , value("data", vistk::datum::data)
        , value("empty", vistk::datum::empty)
        , value("complete", vistk::datum::complete)
        , value("error", vistk::datum::error)
        ]
      .def("type", &vistk::datum::type)
      .def("get_error", &vistk::datum::get_error)
      /// \todo How to do this?
      //.def("get_datum", &vistk::datum::get_datum)
    ]
  ];

  return 0;
}
