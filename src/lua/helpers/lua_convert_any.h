/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_LUA_HELPERS_LUA_CONVERT_ANY_H
#define VISTK_LUA_HELPERS_LUA_CONVERT_ANY_H

#include "lua_include.h"

#include <luabind/object.hpp>
#include <luabind/detail/class_rep.hpp>
#include <luabind/detail/policy.hpp>

#include <boost/any.hpp>
#include <boost/function.hpp>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

/**
 * \file lua_convert_any.h
 *
 * \brief Helpers for working with boost::any in Lua.
 */

class any_converter
{
  public:
    typedef boost::function<boost::any (lua_State*, int)> from_converter_t;
    typedef boost::function<void (lua_State*, boost::any const&)> to_converter_t;

    any_converter();
    ~any_converter();

    template <typename T>
    void register_type();

    from_converter_t from_converter(luabind::type_id const& type) const;
    to_converter_t to_converter(luabind::type_id const& type) const;
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

typedef boost::shared_ptr<any_converter> any_converter_t;

void register_any_coverters(lua_State* L);

extern std::string const vistk_any_converter_variable;
bool is_class_instance(lua_State* L, int index);

namespace luabind
{

template <>
struct default_converter<boost::any>
  : native_converter_base<boost::any>
{
  public:
    static int compute_score(lua_State* /*L*/, int /*index*/)
    {
      return 0;
    }

    boost::any from(lua_State* L, int index)
    {
      boost::optional<any_converter_t> const convs = luabind::object_cast_nothrow<any_converter_t>(registry(L)[vistk_any_converter_variable]);

      if (!convs || !*convs)
      {
        /// \todo Log a warning that the converters have not been registered.
        return boost::any();
      }

      if (is_class_instance(L, index))
      {
        /// \todo How to get the converter for this?
      }
      else
      {
#define TRY_CONVERT_FROM_RAW(T)                                        \
  if (!luabind::default_converter<T>().compute_score(L, index))        \
  {                                                                    \
    return boost::any(luabind::default_converter<T>().from(L, index)); \
  }
#define TRY_CONVERT_FROM(T)          \
  TRY_CONVERT_FROM_RAW(T)            \
  else TRY_CONVERT_FROM_RAW(T const) \
  else TRY_CONVERT_FROM_RAW(T const&)

        TRY_CONVERT_FROM(bool)
        else TRY_CONVERT_FROM(char)
        else TRY_CONVERT_FROM(char signed)
        else TRY_CONVERT_FROM(char unsigned)
        else TRY_CONVERT_FROM(short signed)
        else TRY_CONVERT_FROM(short unsigned)
        else TRY_CONVERT_FROM(int signed)
        else TRY_CONVERT_FROM(int unsigned)
        else TRY_CONVERT_FROM(long signed)
        else TRY_CONVERT_FROM(long unsigned)
        else TRY_CONVERT_FROM(float)
        else TRY_CONVERT_FROM(double)
        else TRY_CONVERT_FROM(long double)
        else TRY_CONVERT_FROM(std::string)

#undef TRY_CONVERT_FROM
#undef TRY_CONVERT_FROM_RAW
      }

      return boost::any();
    }

    void to(lua_State* L, boost::any const& a)
    {
      any_converter_t const convs = luabind::object_cast<any_converter_t>(registry(L)[vistk_any_converter_variable]);

      if (!convs)
      {
        /// \todo Log a warning that the converters have not been registered.
        lua_pushnil(L);
        return;
      }

      any_converter::to_converter_t const conv = convs->to_converter(a.type());

      if (!conv)
      {
        /// \todo Log a warning that the requested type has not been registered.
        lua_pushnil(L);
        return;
      }

      conv(L, a);
    }
};

template <>
struct default_converter<boost::any const&>
  : native_converter_base<boost::any>
{
};

}

#endif // VISTK_LUA_HELPERS_LUA_CONVERT_ANY_H
