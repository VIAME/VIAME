/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <lua/helpers/lua_include.h>

#include <vistk/lua/any_conversion/prototypes.h>
#include <vistk/lua/any_conversion/registration.h>

#include <vistk/pipeline/datum.h>

#include <luabind/class.hpp>
#include <luabind/function.hpp>

#include <boost/any.hpp>
#include <boost/cstdint.hpp>

#include <string>

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

static vistk::datum_t new_datum(object const& obj);

int
luaopen_vistk_pipeline_datum(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline")
    [
      class_<vistk::datum::error_t>("datum_error")
        .def(constructor<>())
    , def("new_datum", &new_datum)
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
      .def("get_datum", &vistk::datum::get_datum<boost::any>)
    ]
  ];

  vistk::lua::register_type<std::string>(0);
  vistk::lua::register_type<int32_t>(1);
  vistk::lua::register_type<char>(2);
  vistk::lua::register_type<bool>(3);
  vistk::lua::register_type<double>(4);

  return 0;
}

vistk::datum_t
new_datum(object const& obj)
{
  return vistk::datum::new_datum(object_cast<boost::any>(obj));
}
