/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "lua_convert_any.h"

extern "C"
{
#include <lua.h>
}

#include <luabind/class.hpp>

#include <boost/any.hpp>
#include <boost/make_shared.hpp>

#include <map>
#include <string>

/**
 * \file datum.cxx
 *
 * \brief Lua bindings for \link vistk::datum\endlink.
 */

class any_converter::priv
{
  public:
    priv();
    ~priv();

    template <typename T>
    struct convert
    {
      public:
        static boost::any from(lua_State* L, int index);
        static void to(lua_State* L, boost::any const& a);
    };

    typedef std::map<luabind::type_id, from_converter_t> from_any_converters_t;
    typedef std::map<luabind::type_id, to_converter_t> to_any_converters_t;

    from_any_converters_t from_converters;
    to_any_converters_t to_converters;
};

any_converter
::any_converter()
  : d(new priv)
{
}

any_converter
::~any_converter()
{
}

any_converter::priv
::priv()
{
}

any_converter::priv
::~priv()
{
}

static void make_any_coverters(any_converter_t conv);

std::string const vistk_any_converter_variable = "_vistk_any_converter";

void
register_any_coverters(lua_State* L)
{
  luabind::module(L, "vistk")
  [
    luabind::namespace_("_private_any")
    [
      luabind::class_<any_converter, any_converter_t>("any_converter")
    ]
  ];

  any_converter_t conv = boost::make_shared<any_converter>();

  make_any_coverters(conv);

  luabind::registry(L)[vistk_any_converter_variable] = conv;
}

void
make_any_coverters(any_converter_t conv)
{
  conv->register_type<int>();
  conv->register_type<float>();
  conv->register_type<std::string>();
}

template <typename T>
void
any_converter
::register_type()
{
  d->from_converters[luabind::type_id(typeid(T))] = &priv::convert<T>::from;
  d->to_converters[luabind::type_id(typeid(T))] = &priv::convert<T>::to;
}

any_converter::from_converter_t
any_converter
::from_converter(luabind::type_id const& type) const
{
  priv::from_any_converters_t::const_iterator const i = d->from_converters.find(type);

  if (i == d->from_converters.end())
  {
    return from_converter_t();
  }

  return i->second;
}

any_converter::to_converter_t
any_converter
::to_converter(luabind::type_id const& type) const
{
  priv::to_any_converters_t::const_iterator const i = d->to_converters.find(type);

  if (i == d->to_converters.end())
  {
    return to_converter_t();
  }

  return i->second;
}

template <typename T>
boost::any
any_converter::priv::convert<T>
::from(lua_State* L, int index)
{
  luabind::object const obj(luabind::from_stack(L, index));

  return boost::any(*luabind::touserdata<T>(obj));
}

template <typename T>
void
any_converter::priv::convert<T>
::to(lua_State* L, boost::any const& a)
{
  luabind::detail::convert_to_lua(L, *boost::any_cast<T>(&a));
}

template <>
void
any_converter::priv::convert<void>
::to(lua_State* L, boost::any const& /*a*/)
{
  lua_pushnil(L);
}

bool
is_class_instance(lua_State* L, int index)
{
  if (!lua_getmetatable(L, index))
  {
    return false;
  }

  lua_pushstring(L, "__luabind_class");
  lua_gettable(L, -2);

  bool is_class = false;

  if (lua_toboolean(L, -1))
  {
    is_class = true;
  }

  lua_pop(L, 2);

  return is_class;
}
