/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_LUA_PIPELINE_UTIL_LUA_CONVERT_OPTIONAL_H
#define VISTK_LUA_PIPELINE_UTIL_LUA_CONVERT_OPTIONAL_H

#include "lua_include.h"

#include <luabind/detail/policy.hpp>
#include <luabind/object.hpp>

#include <boost/optional.hpp>

/**
 * \file lua_convert_optional.h
 *
 * \brief Helpers for working with boost::optional in Lua.
 */

namespace luabind
{

template <typename T>
struct default_converter<boost::optional<T> >
  : native_converter_base<boost::optional<T> >
{
  public:
    static int compute_score(lua_State* L, int index)
    {
      object const obj(from_stack(L, index));

      return (object_cast_nothrow<T>(obj) ? 0 : -1);

    }

    boost::optional<T> from(lua_State* L, int index)
    {
      object const obj(from_stack(L, index));

      return object_cast<T>(obj);
    }

    void to(lua_State* L, boost::optional<T> const& o)
    {
      if (o)
      {
        detail::convert_to_lua(L, *o);
      }
      else
      {
        lua_pushnil(L);
      }
    }
};

template <typename T>
struct default_converter<boost::optional<T>&>
  : default_converter<boost::optional<T> >
{
};

template <typename T>
struct default_converter<boost::optional<T> const&>
  : default_converter<boost::optional<T> >
{
};

}

#endif // VISTK_LUA_PIPELINE_UTIL_LUA_CONVERT_OPTIONAL_H
