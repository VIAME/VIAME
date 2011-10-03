# - Find lua interpreter
# This module finds if Lua interpreter is installed and determines where the
# executables are. This code sets the following variables:
#
#  LUAINTERP_FOUND         - Was the Lua executable found
#  LUA_EXECUTABLE          - path to the Lua interpreter
#  Lua_ADDITIONAL_VERSIONS - list of additional Lua versions to search for
#

#=============================================================================
# Copyright 2005-2010 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

# Set up the versions we know about, in the order we will search. Always add
# the user supplied additional versions to the front.
set(_Lua_VERSIONS
  ${Lua_ADDITIONAL_VERSIONS}
  5.1 5.0)

# Run first with the Lua version in the executable
foreach(_CURRENT_VERSION ${_Lua_VERSIONS})
set(_Lua_NAMES lua${_CURRENT_VERSION})
  if(WIN32)
    list(APPEND _Lua_NAMES lua)
  endif()
  find_program(LUA_EXECUTABLE
    NAMES ${_Lua_NAMES}
    )
endforeach()
# Now without any version if we still haven't found it
if(NOT LUA_EXECUTABLE)
  find_program(LUA_EXECUTABLE NAMES lua)
endif()


# handle the QUIETLY and REQUIRED arguments and set LUAINTERP_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LuaInterp DEFAULT_MSG LUA_EXECUTABLE)

mark_as_advanced(LUA_EXECUTABLE)
