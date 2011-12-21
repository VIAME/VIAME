/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_LUA_ANY_CONVERSION_PROTOTYPES_H
#define VISTK_LUA_ANY_CONVERSION_PROTOTYPES_H

#include "registration.h"

#include <luabind/detail/policy.hpp>

/**
 * \file any_conversion/prototypes.h
 *
 * \brief Prototype functions for converting types to and from boost::any for Lua.
 */

namespace vistk
{

namespace lua
{

/**
 * \brief Converts a Lua object into a boost::any.
 *
 * \param obj The object to convert.
 * \param storage The memory location to construct the object.
 *
 * \return True if the conversion succeeded, false otherwise.
 */
template <typename T>
opt_any_t
from_prototype(lua_State* L, int index)
{
  luabind::default_converter<T> conv = luabind::default_converter<T>();

  if (!conv.compute_score(L, index))
  {
    return boost::any(conv.from(L, index));
  }

  return opt_any_t();
}

/**
 * \brief Converts a boost::any into a Lua object.
 *
 * \param any The object to convert.
 *
 * \return The object if created, nothing otherwise.
 */
template <typename T>
bool
to_prototype(lua_State* L, boost::any const& any)
{
  try
  {
    T const t = boost::any_cast<T>(any);
    luabind::default_converter<T>().to(L, t);
    return true;
  }
  catch (boost::bad_any_cast&)
  {
  }

  return false;
}

/**
 * \brief Registers a type for conversion between Lua and boost::any.
 *
 * \param priority The priority for the type conversion.
 */
template <typename T>
void
register_type(priority_t priority)
{
  register_conversion(priority, from_prototype<T>, to_prototype<T>);
}

}

}

#endif // VISTK_LUA_ANY_CONVERSION_PROTOTYPES_H
