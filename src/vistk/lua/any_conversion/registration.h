/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_LUA_ANY_CONVERSION_REGISTRATION_H
#define VISTK_LUA_ANY_CONVERSION_REGISTRATION_H

#include "any_conversion-config.h"

extern "C"
{
#include <lua.h>
#include <lualib.h>
}

#include <luabind/detail/policy.hpp>

#include <boost/any.hpp>
#include <boost/cstdint.hpp>
#include <boost/function.hpp>
#include <boost/optional.hpp>

/**
 * \file any_conversion/registration.h
 *
 * \brief Helpers for working with boost::any in Lua.
 */

namespace vistk
{

namespace lua
{

/// A type for a possible Lua conversion.
typedef boost::optional<boost::any> opt_any_t;

/// A function which converts from Lua, returning \c true on success.
typedef boost::function<opt_any_t (lua_State*, int)> from_any_func_t;
/// A function which converts to Lua, returning the \c true on success.
typedef boost::function<bool (lua_State*, boost::any const&)> to_any_func_t;

/// A priority for converting between boost::any and Lua.
typedef uint64_t priority_t;

/**
 * \brief Registers functions for conversions between boost::any and Lua.
 *
 * \param priority The priority for the type conversion.
 * \param from The function for converting from Lua.
 * \param to The function for converting to Lua.
 */
void VISTK_LUA_ANY_CONVERSION_EXPORT register_conversion(priority_t priority, from_any_func_t from, to_any_func_t to);

}

}

namespace luabind
{

/**
 * \struct default_converter "registration.h" <vistk/lua/any_conversion/registration.h>
 *
 * \brief The converter for boost::any for Lua.
 */
template <>
struct default_converter<boost::any>
  : native_converter_base<boost::any>
{
  public:
    /**
     * \brief Determines the "strength" of a possible conversion.
     *
     * \param L The Lua interpreter state.
     * \param index
     *
     * \return \c 0 for compatible, negative for incompatible.
     */
    static int compute_score(lua_State* L, int index);

    /**
     * \brief Converts a Lua object into a boost::any.
     *
     * \param L The Lua interpreter state.
     * \param index The index of the object to convert.
     *
     * \return The converted boost::any.
     */
    boost::any from(lua_State* L, int index);
    /**
     * \brief Converts a boost::any into a Lua object.
     *
     * \param L The Lua interpreter state.
     * \param any The boost::any to convert.
     */
    void to(lua_State* L, boost::any const& any);
};

/**
 * \struct default_converter "registration.h" <vistk/lua/any_conversion/registration.h>
 *
 * \brief The converter for boost::any const for Lua.
 */
template <>
struct default_converter<boost::any const>
  : default_converter<boost::any>
{
};

/**
 * \struct default_converter "registration.h" <vistk/lua/any_conversion/registration.h>
 *
 * \brief The converter for boost::any const& for Lua.
 */
template <>
struct default_converter<boost::any const&>
  : default_converter<boost::any>
{
};

}

#endif // VISTK_LUA_ANY_CONVERSION_REGISTRATION_H
