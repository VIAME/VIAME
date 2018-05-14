#ckwg +4
# Copyright 2012-2014 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

# Locate the system installed CppDB
#
# The following variables will guide the build:
#
# CppDB_ROOT        - Set to the install prefix of the CppDB library
#
# The following variables will be set:
#
# CppDB_FOUND       - Set to true if CppDB can be found
# CppDB_INCLUDE_DIR - The path to the CppDB header files
# CppDB_LIBRARY     - The full path to the CppDB library
# CppDB_LIB_DIR

if( CppDB_DIR )
  find_package( CppDB NO_MODULE )
elseif( NOT CppDB_FOUND )
  include(CommonFindMacros)

  setup_find_root_context(CppDB)
  find_path( CppDB_INCLUDE_DIR cppdb/driver_manager.h ${CppDB_FIND_OPTS})
  find_library( CppDB_LIBRARY cppdb ${CppDB_FIND_OPTS})
  restore_find_root_context(CppDB)

  include( FindPackageHandleStandardArgs )
  FIND_PACKAGE_HANDLE_STANDARD_ARGS( CppDB CppDB_INCLUDE_DIR CppDB_LIBRARY)
  if(CPPDB_FOUND)
    set(CppDB_FOUND True)
    get_filename_component(CppDB_LIB_DIR ${CppDB_LIBRARY} PATH)
  endif()
endif()
