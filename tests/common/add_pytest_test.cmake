# Shared helper for registering pytest-based VIAME tests as ctest tests.
#
# examples, pipelines, and the pytorch plugin tests register pytest classes as
# ctest tests with largely the same plumbing. This centralizes it so the
# per-subtree CMakeLists only describe what differs.
#
# Two execution models, selected by SOURCE_SETUP:
#   default      - run python -m pytest with PYTHONPATH / VIAME_INSTALL set via
#                  the test ENVIRONMENT (examples, pipelines).
#   SOURCE_SETUP - source setup_viame.{sh,bat} first, then run pytest in the
#                  same shell, for plugin tests that import native modules
#                  needing the full sourced environment.

# Directory holding the shared python helpers (tests/common); always added to
# PYTHONPATH (default mode) so test modules can `from viame_env import ...`.
set( VIAME_TESTS_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}" )

if( NOT Python3_EXECUTABLE )
  find_package( Python3 COMPONENTS Interpreter REQUIRED )
endif()

if( WIN32 )
  set( VIAME_TEST_PYPATH_SEP "\;" )
else()
  set( VIAME_TEST_PYPATH_SEP ":" )
endif()

# viame_add_pytest_test(
#   NAME <ctest name>
#   TARGET <pytest args...>           # e.g. <file.py> -k <Class>  OR  <file.py>::<Class>
#   [LABELS <label>...]
#   [TIMEOUT <seconds>]
#   [WORKING_DIRECTORY <dir>]
#   [SKIP_RETURN_CODE <code>]
#   [PYTHONPATH_DIRS <dir>...]        # extra dirs prepended to PYTHONPATH (default mode)
#   [SOURCE_SETUP]                    # source setup_viame.{sh,bat} before pytest
#   [DISABLED] )
function( viame_add_pytest_test )
  set( options DISABLED SOURCE_SETUP )
  set( oneValueArgs NAME TIMEOUT WORKING_DIRECTORY SKIP_RETURN_CODE )
  set( multiValueArgs TARGET LABELS PYTHONPATH_DIRS )
  cmake_parse_arguments( PT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  set( install_dir "${VIAME_BUILD_INSTALL_PREFIX}" )

  if( PT_SOURCE_SETUP )
    # Source the install setup script, then run pytest in the same shell so
    # native modules resolve. The test module self-paths tests/common.
    string( JOIN " " target_str ${PT_TARGET} )
    if( WIN32 )
      set( setup_cmd "call \"${install_dir}/setup_viame.bat\"" )
    else()
      set( setup_cmd "source \"${install_dir}/setup_viame.sh\"" )
    endif()
    add_test(
      NAME "${PT_NAME}"
      COMMAND bash -c "${setup_cmd} && python -m pytest ${target_str} -v --tb=short"
    )
  else()
    set( py_path "${install_dir}/python" )
    set( site_packages
      "${install_dir}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages" )

    set( pythonpath_parts "${py_path}" "${site_packages}" "${VIAME_TESTS_COMMON_DIR}" )
    foreach( extra_dir IN LISTS PT_PYTHONPATH_DIRS )
      list( APPEND pythonpath_parts "${extra_dir}" )
    endforeach()
    string( JOIN "${VIAME_TEST_PYPATH_SEP}" pythonpath ${pythonpath_parts} )

    add_test(
      NAME "${PT_NAME}"
      COMMAND ${Python3_EXECUTABLE} -m pytest ${PT_TARGET} -v --tb=short
    )
  endif()

  if( PT_LABELS )
    set_property( TEST "${PT_NAME}" PROPERTY LABELS ${PT_LABELS} )
  endif()

  if( NOT PT_SOURCE_SETUP )
    set_property( TEST "${PT_NAME}" PROPERTY ENVIRONMENT
            "PYTHONPATH=${pythonpath}${VIAME_TEST_PYPATH_SEP}$ENV{PYTHONPATH}"
            "VIAME_INSTALL=${install_dir}"
    )
  endif()

  if( PT_TIMEOUT )
    set_property( TEST "${PT_NAME}" PROPERTY TIMEOUT "${PT_TIMEOUT}" )
  endif()

  if( PT_WORKING_DIRECTORY )
    set_property( TEST "${PT_NAME}" PROPERTY WORKING_DIRECTORY "${PT_WORKING_DIRECTORY}" )
  endif()

  if( NOT "${PT_SKIP_RETURN_CODE}" STREQUAL "" )
    set_property( TEST "${PT_NAME}" PROPERTY SKIP_RETURN_CODE "${PT_SKIP_RETURN_CODE}" )
  endif()

  if( PT_DISABLED )
    set_property( TEST "${PT_NAME}" PROPERTY DISABLED TRUE )
  endif()
endfunction()
