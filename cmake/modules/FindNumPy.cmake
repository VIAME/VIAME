#ckwg +4
# Copyright 2010, 2012, 2013 by Kitware, Inc. All Rights Reserved. Please refer
# to KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

# Locate NumPy library
# This module defines
#  NUMPY_FOUND       - if false, do not try to link to NumPy
#  NUMPY_INCLUDE_DIR - where to find numpy/numpyconfig.h
#  NUMPY_VERSION     - the version of NumPy

if (PYTHON_EXECUTABLE)
  execute_process(
    COMMAND         "${PYTHON_EXECUTABLE}" -c
                    "import numpy, sys; sys.stdout.write(numpy.get_include())"
    RESULT_VARIABLE __numpy_include_dir_res
    OUTPUT_VARIABLE __numpy_include_dir
    ERROR_QUIET)

  if (NOT __numpy_include_dir_res)
    set(NUMPY_INCLUDE_DIR "${__numpy_include_dir}"
      CACHE PATH "Include directory for NumPy headers")
  endif ()

  execute_process(
    COMMAND         "${PYTHON_EXECUTABLE}" -c
                    "import numpy, sys; sys.stdout.write(numpy.version.version)"
    RESULT_VARIABLE __numpy_version_res
    OUTPUT_VARIABLE __numpy_version
    ERROR_QUIET)

  if (${__numpy_version_res} EQUAL 0)
    set(NUMPY_VERSION "${__numpy_version}"
      CACHE PATH "NumPy version")
  endif ()
endif ()

# handle the QUIETLY and REQUIRED arguments and set NUMPY_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy
  REQUIRED_VARS
    NUMPY_INCLUDE_DIR
  VERSION_VAR
    NUMPY_VERSION)

mark_as_advanced(
  NUMPY_INCLUDE_DIR)
