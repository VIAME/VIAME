#ckwg +4
# Copyright 2010 2014 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

# Locate the system installed Log4cplus
# The following variables will be set:
#
# Log4cplus_FOUND       - Set to true if Log4cplus can be found
# Log4cplus_INCLUDE_DIR - The path to the Log4cplus header files
# Log4cplus_LIBRARY     - The full path to the Log4cplus library

if( Log4cplus_DIR )
  find_package( Log4cplus NO_MODULE )
elseif( NOT Log4cplus_FOUND )
  include(CommonFindMacros)

  setup_find_root_context(Log4cplus)
  find_path( Log4cplus_INCLUDE_DIR log4cplus/logger.h ${Log4cplus_FIND_OPTS})
  find_library( Log4cplus_LIBRARY log4cplus ${Log4cplus_FIND_OPTS})
  restore_find_root_context(Log4cplus)

  include( FindPackageHandleStandardArgs )
  FIND_PACKAGE_HANDLE_STANDARD_ARGS( Log4cplus Log4cplus_INCLUDE_DIR Log4cplus_LIBRARY )
  if( LOG4CPLUS_FOUND )
    set( Log4cplus_FOUND TRUE )
  endif()
endif()
