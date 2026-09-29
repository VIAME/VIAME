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

# The interpreter, and its version, for every test registered here.
#
# The tree configures python with `find_package( Python )`, which populates
# `Python_*` and leaves `Python3_*` empty. Taking `Python3_EXECUTABLE` alone
# and searching for the rest went wrong twice: a bare
# `find_package( Python3 )` picks the newest interpreter on PATH rather than
# the one the tree was built against, and `Python3_VERSION_MAJOR` being empty
# built a site-packages path of `lib/python./site-packages`, which silently
# dropped pytest and everything else off PYTHONPATH.
#
# So prefer what the tree already found, fall back to a Python3 search only
# when there is nothing, and refuse to register tests against a version we
# could not determine rather than emit a path with a hole in it.
if( Python_EXECUTABLE AND Python_VERSION_MAJOR )
  set( VIAME_TEST_PYTHON "${Python_EXECUTABLE}" )
  set( VIAME_TEST_PYTHON_VERSION
       "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}" )
elseif( Python3_EXECUTABLE AND Python3_VERSION_MAJOR )
  set( VIAME_TEST_PYTHON "${Python3_EXECUTABLE}" )
  set( VIAME_TEST_PYTHON_VERSION
       "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}" )
else()
  find_package( Python3 COMPONENTS Interpreter REQUIRED )
  set( VIAME_TEST_PYTHON "${Python3_EXECUTABLE}" )
  set( VIAME_TEST_PYTHON_VERSION
       "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}" )
endif()

if( NOT VIAME_TEST_PYTHON_VERSION MATCHES "^[0-9]+\\.[0-9]+$" )
  message( FATAL_ERROR
    "Could not determine the python version for the tests; got "
    "'${VIAME_TEST_PYTHON_VERSION}' from '${VIAME_TEST_PYTHON}'" )
endif()

if( WIN32 )
  set( VIAME_TEST_PYPATH_SEP "\;" )
else()
  set( VIAME_TEST_PYPATH_SEP ":" )
endif()

# viame_add_test(
#   NAME <ctest name>
#   TARGET <pytest args...>           # e.g. <file.py> -k <Class>  OR  <file.py>::<Class>
#   [LABELS <label>...]
#   [TIMEOUT <seconds>]
#   [WORKING_DIRECTORY <dir>]
#   [SKIP_RETURN_CODE <code>]
#   [PYTHONPATH_DIRS <dir>...]        # extra dirs prepended to PYTHONPATH (default mode)
#   [SOURCE_SETUP]                    # source setup_viame.{sh,bat} before pytest
#   [DISABLED] )
function( viame_add_test )
  set( options DISABLED SOURCE_SETUP )
  set( oneValueArgs NAME TIMEOUT WORKING_DIRECTORY SKIP_RETURN_CODE )
  set( multiValueArgs TARGET LABELS PYTHONPATH_DIRS )
  cmake_parse_arguments( PT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  set( install_dir "${VIAME_BUILD_INSTALL_PREFIX}" )

  if( PT_SOURCE_SETUP )
    # Source the install setup script, then run pytest in the same shell so
    # native modules resolve. The test module self-paths tests/common.
    string( JOIN " " target_str ${PT_TARGET} )
    # The shell has to match the script. `call` and a `.bat` are cmd's, and
    # running them under bash -- which every test did -- fails with
    # "call: command not found" before pytest is reached.
    #
    # A generated script rather than `cmd /c "call ... && ..."`: ctest quotes
    # each argument the way the C runtime parses them, cmd parses quotes its
    # own way, and the two disagree over the inner pair ("The syntax of the
    # command is incorrect"). One path argument has nothing to disagree about.
    if( WIN32 )
      string( MAKE_C_IDENTIFIER "${PT_NAME}" _pt_slug )
      set( _pt_runner "${CMAKE_CURRENT_BINARY_DIR}/run_${_pt_slug}.bat" )
      file( TO_NATIVE_PATH "${install_dir}/setup_viame.bat" _pt_setup )
      file( WRITE "${_pt_runner}"
"@echo off
call \"${_pt_setup}\"
python -m pytest ${target_str} -v --tb=short
" )
      # Through `cmake -E env`, so that cmd.exe is not the test's own
      # executable. cmd re-parses its command line including the program
      # name and reads a `/` in it as the start of a switch; CMake stores a
      # test's executable with forward slashes, so `COMMAND cmd /c ...`
      # reaches cmd as `C:/Windows/System32/cmd.exe ...` and it answers
      # "The syntax of the command is incorrect" without looking at the
      # script. As an argument to `cmake -E env`, `cmd` stays a bare name.
      file( TO_NATIVE_PATH "${_pt_runner}" _pt_runner_native )
      add_test(
        NAME "${PT_NAME}"
        COMMAND ${CMAKE_COMMAND} -E env cmd /c "${_pt_runner_native}"
      )
    else()
      add_test(
        NAME "${PT_NAME}"
        COMMAND bash -c "source \"${install_dir}/setup_viame.sh\" && python -m pytest ${target_str} -v --tb=short"
      )
    endif()
  else()
    set( py_path "${install_dir}/python" )
    # Windows python has no version level in its layout; see the WIN32
    # branch of `viame_python_install_path` in viame_project.cmake.
    if( WIN32 )
      set( site_packages "${install_dir}/Lib/site-packages" )
    else()
      set( site_packages
        "${install_dir}/lib/python${VIAME_TEST_PYTHON_VERSION}/site-packages" )
    endif()

    set( pythonpath_parts "${py_path}" "${site_packages}" "${VIAME_TESTS_COMMON_DIR}" )
    foreach( extra_dir IN LISTS PT_PYTHONPATH_DIRS )
      list( APPEND pythonpath_parts "${extra_dir}" )
    endforeach()
    string( JOIN "${VIAME_TEST_PYPATH_SEP}" pythonpath ${pythonpath_parts} )

    add_test(
      NAME "${PT_NAME}"
      COMMAND ${VIAME_TEST_PYTHON} -m pytest ${PT_TARGET} -v --tb=short
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
