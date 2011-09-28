#ckwg +4
# Copyright 2010 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

# Locate Luabind library
# This module defines
#  LUABIND_FOUND       - if false, do not try to link to Luabind
#  LUABIND_INCLUDE_DIR - where to find luabind/luabind.hpp
#  LUABIND_LIBRARY     - the full path of the luabind library

find_path(LUABIND_INCLUDE_DIR luabind/luabind.hpp)

find_library(LUABIND_LIBRARY luabind)

#include(FindPackageHandleStandardArgs.cmake)
# handle the QUIETLY and REQUIRED arguments and set LUA_FOUND to TRUE if
# all listed variables are TRUE
#find_package_handle_standard_args(Luabind DEFAULT_MSG LUABIND_LIBRARY LUABIND_INCLUDE_DIR)

mark_as_advanced(
  LUABIND_INCLUDE_DIR
  LUABIND_LIBRARY)
