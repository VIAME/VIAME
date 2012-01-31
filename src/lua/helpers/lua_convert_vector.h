/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_LUA_HELPERS_LUA_CONVERT_VECTOR_H
#define VISTK_LUA_HELPERS_LUA_CONVERT_VECTOR_H

#include "lua_include.h"

#include <luabind/object.hpp>

#include <boost/foreach.hpp>

#include <vector>

/**
 * \file lua_convert_vector.h
 *
 * \brief Helpers for working with std::vector in Lua.
 */

namespace luabind
{

template <typename T>
struct default_converter<std::vector<T> >
  : native_converter_base<std::vector<T> >
{
  public:
    static int compute_score(lua_State* L, int index)
    {
      return ((lua_type(L, index) == LUA_TTABLE) ? 0 : -1);
    }

    std::vector<T> from(lua_State* L, int index)
    {
      std::vector<T> vec;

      object const obj(from_stack(L, index));
      iterator const end;

      for (iterator i(obj); i != end; ++i)
      {
        vec.push_back(object_cast<T>(*i));
      }

      return vec;
    }

    void to(lua_State* L, std::vector<T> const& v)
    {
      object table = newtable(L);

      size_t i = 0;

      BOOST_FOREACH (T const& e, v)
      {
        table[i++] = boost::cref(e);
      }

      table.push(L);
    }
};

template <typename T>
struct default_converter<std::vector<T> const&>
  : default_converter<std::vector<T> >
{
};

}

#endif // VISTK_LUA_HELPERS_LUA_CONVERT_VECTOR_H
