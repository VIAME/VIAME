/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_LUA_PIPELINE_LUA_STATIC_MEMBER_H
#define VISTK_LUA_PIPELINE_LUA_STATIC_MEMBER_H

#include "lua_include.h"

#include <cstring>

/**
 * \file lua_static_member.h
 *
 * \brief Helpers setting static members on objects in Lua.
 */

#define LUA_STATIC_boolean(var)   var
#define LUA_STATIC_cfunction(var) &var
#define LUA_STATIC_integer(var)   var
#define LUA_STATIC_literal(var)   var
#define LUA_STATIC_lstring(var)   var, strlen(var)
#define LUA_STATIC_number(var)    var
#define LUA_STATIC_string(var)    var.c_str()

#define LUA_STATIC_MEMBER(interp, type, var, name)  \
  do                                                \
  {                                                 \
    lua_push##type(interp, LUA_STATIC_##type(var)); \
    lua_setfield(L, -2, name);                      \
  } while (false)

#endif // VISTK_LUA_PIPELINE_LUA_STATIC_MEMBER_H
