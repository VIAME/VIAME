/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_LUA_HELPERS_LUA_CONVERT_SET_H
#define VISTK_LUA_HELPERS_LUA_CONVERT_SET_H

#include "lua_include.h"

#include <luabind/luabind.hpp>

#include <boost/foreach.hpp>

#include <set>

/**
 * \file lua_convert_set.h
 *
 * \brief Helpers for working with std::set in Lua.
 */

namespace luabind
{

template <typename T>
struct default_converter<std::set<T> >
  : native_converter_base<std::set<T> >
{
  public:
    static int compute_score(lua_State* L, int index)
    {
      return lua_type(L, index) == LUA_TTABLE ? 0 : -1;
    }

    std::set<T> from(lua_State* L, int index)
    {
      std::set<T> s;

      object const obj(from_stack(L, index));
      iterator const end;

      for (iterator i(obj); i != end; ++i)
      {
        s.insert(object_cast<T>(*i));
      }

      return s;
    }

    void to(lua_State* L, std::set<T> const& s)
    {
      object table = newtable(L);

      size_t i = 0;

      BOOST_FOREACH (T const& e, s)
      {
        table[i++] = boost::cref(e);
      }

      table.push(L);
    }
};

template <typename T>
struct default_converter<std::set<T> const&>
  : default_converter<std::set<T> >
{
};

}

#endif // VISTK_LUA_HELPERS_LUA_CONVERT_SET_H
