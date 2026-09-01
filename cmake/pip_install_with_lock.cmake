# CMake script to run pip install with file locking to prevent race conditions
#
# Parameters (choose one mode):
#   Mode 1 - Wheel install:
#     WHEEL_DIR - Directory containing .whl files to install
#     FORCE_REINSTALL - If TRUE, uses --force-reinstall --no-deps (for rebuilds)
#
#   Mode 2 - Direct args:
#     PIP_ARGS - Arguments to pass to pip install (separated by ----)
#
# Common parameters:
#   Python_EXECUTABLE - Path to python executable
#   WORKING_DIR - Optional working directory
#   NO_CACHE_DIR - If set to TRUE, adds --no-cache-dir to pip command
#   ENV_VARS - Environment variables (----separated KEY=VALUE pairs, <PS> for path separator)

cmake_minimum_required( VERSION 3.16 )

# Build the pip install arguments
if( WHEEL_DIR )
  # Mode 1: Install wheels from directory (locally built wheels)
  file( GLOB _all_wheels LIST_DIRECTORIES FALSE "${WHEEL_DIR}/*.whl" )

  # Check if any wheels were found
  list( LENGTH _all_wheels _wheel_count )
  if( _wheel_count EQUAL 0 )
    message( FATAL_ERROR "No wheel files found in WHEEL_DIR: ${WHEEL_DIR}" )
  endif()

  # When multiple wheels exist (e.g., platform-specific and pure-python),
  # prefer platform-specific wheels (cpXX-cpXX-platform) over py3-none-any
  set( _platform_wheels )
  set( _pure_wheels )
  foreach( _wheel IN LISTS _all_wheels )
    if( _wheel MATCHES "-cp[0-9]+-cp[0-9]+-" )
      list( APPEND _platform_wheels "${_wheel}" )
    else()
      list( APPEND _pure_wheels "${_wheel}" )
    endif()
  endforeach()

  # Use platform-specific wheels if available, otherwise use all wheels
  if( _platform_wheels )
    set( _candidate_wheels "${_platform_wheels}" )
  else()
    set( _candidate_wheels "${_all_wheels}" )
  endif()

  # A wheel dir accumulates every version ever built into it, and handing pip
  # two versions of one distribution is an unsatisfiable request rather than a
  # newest-wins choice. Keep only the most recently written wheel per
  # distribution, so a version bump does not collide with its predecessor.
  # Distributions other than each other are all still installed together.
  set( _dist_names )
  set( _pip_args )
  foreach( _wheel IN LISTS _candidate_wheels )
    get_filename_component( _wheel_file "${_wheel}" NAME )
    string( REGEX REPLACE "^([^-]+)-.*$" "\\1" _dist "${_wheel_file}" )
    file( TIMESTAMP "${_wheel}" _wheel_time "%Y%m%d%H%M%S" )

    list( FIND _dist_names "${_dist}" _dist_index )
    if( _dist_index EQUAL -1 )
      list( APPEND _dist_names "${_dist}" )
      list( APPEND _pip_args "${_wheel}" )
      set( _dist_time_${_dist} "${_wheel_time}" )
    elseif( _wheel_time STRGREATER "${_dist_time_${_dist}}" )
      list( REMOVE_AT _pip_args ${_dist_index} )
      list( INSERT _pip_args ${_dist_index} "${_wheel}" )
      set( _dist_time_${_dist} "${_wheel_time}" )
    endif()
  endforeach()

  list( LENGTH _candidate_wheels _candidate_count )
  list( LENGTH _pip_args _selected_count )
  if( NOT _candidate_count EQUAL _selected_count )
    message( STATUS
      "Ignoring superseded wheels in ${WHEEL_DIR}: installing "
      "${_selected_count} of ${_candidate_count}" )
  endif()

  set( _working_dir "${WHEEL_DIR}" )

  # Use force reinstall without deps for rebuilds (not first builds)
  set( _force_flag "" )
  if( FORCE_REINSTALL )
    set( _force_flag "--force-reinstall" "--no-deps" )
  endif()
elseif( PIP_ARGS )
  # Mode 2: Use provided arguments (external packages from PyPI, etc.)
  string( REPLACE "----" ";" _pip_args "${PIP_ARGS}" )
  set( _working_dir "${WORKING_DIR}" )
  # No force reinstall for external packages
  set( _force_flag "" )
else()
  message( FATAL_ERROR "pip_install_with_lock.cmake requires either WHEEL_DIR or PIP_ARGS" )
endif()

# Use /tmp for lock file to ensure all parallel pip installs use the same lock
set( _lock_file "/tmp/viame_pip_install.lock" )

# Build cache flag
if( NO_CACHE_DIR )
  set( _cache_flag "--no-cache-dir" )
else()
  set( _cache_flag "" )
endif()

# Build environment variables for pip (used with cmake -E env)
# For wheel installs, we only need Python-related env vars, not compiler/CUDA paths
# Skip PATH and other semicolon-containing env vars on Windows as they cause issues
# with cmake -E env argument parsing
set( _pip_env_vars )
set( _has_pythonuserbase FALSE )
if( ENV_VARS )
  # Convert ----separated env vars back to list
  string( REPLACE "----" ";" _env_vars_list "${ENV_VARS}" )
  # Convert <PS> path separator to platform-specific separator
  foreach( _env_var IN LISTS _env_vars_list )
    # Skip PATH on Windows for pip install - it's not needed for installing wheels
    # and the semicolon-separated paths cause issues with cmake -E env argument parsing
    if( WIN32 AND _env_var MATCHES "^PATH=" )
      continue()
    endif()
    # Skip any env vars containing <PS> on Windows - these become semicolons which
    # break cmake -E env. These are typically INCLUDE, LIB, and similar compiler vars
    # that are not needed for pip install of pre-built wheels.
    if( WIN32 AND _env_var MATCHES "<PS>" )
      continue()
    endif()
    # On Unix, convert <PS> to colon path separator
    if( NOT WIN32 )
      string( REPLACE "<PS>" ":" _env_var "${_env_var}" )
    endif()
    # Track if PYTHONUSERBASE was provided
    if( _env_var MATCHES "^PYTHONUSERBASE=" )
      set( _has_pythonuserbase TRUE )
    endif()
    list( APPEND _pip_env_vars "${_env_var}" )
  endforeach()
endif()

# CRITICAL: Ensure PYTHONUSERBASE is always set to prevent packages from being
# installed to ~/.local. If not provided in ENV_VARS, inherit from the parent
# process environment.
if( NOT _has_pythonuserbase )
  set( _inherited_userbase "$ENV{PYTHONUSERBASE}" )
  if( _inherited_userbase )
    list( APPEND _pip_env_vars "PYTHONUSERBASE=${_inherited_userbase}" )
  endif()
endif()

if( UNIX )
  # Use flock on Unix to serialize pip installs (5 minute timeout)
  # Always use cmake -E env to ensure PYTHONUSERBASE is properly propagated
  if( _working_dir )
    execute_process(
      COMMAND flock --timeout 300 ${_lock_file}
        ${CMAKE_COMMAND} -E env ${_pip_env_vars}
        ${Python_EXECUTABLE} -m pip install --no-build-isolation --user ${_cache_flag} ${_force_flag} ${_pip_args}
      RESULT_VARIABLE _result
      WORKING_DIRECTORY ${_working_dir}
    )
  else()
    execute_process(
      COMMAND flock --timeout 300 ${_lock_file}
        ${CMAKE_COMMAND} -E env ${_pip_env_vars}
        ${Python_EXECUTABLE} -m pip install --no-build-isolation --user ${_cache_flag} ${_force_flag} ${_pip_args}
      RESULT_VARIABLE _result
    )
  endif()
else()
  # On Windows, add retries for race conditions
  set( _max_retries 5 )
  set( _retry_count 0 )
  set( _result 1 )

  # Build the pip command - always use cmake -E env to ensure PYTHONUSERBASE is propagated
  set( _pip_cmd ${CMAKE_COMMAND} -E env ${_pip_env_vars}
    ${Python_EXECUTABLE} -m pip install --no-build-isolation --user ${_cache_flag} ${_force_flag} ${_pip_args} )

  while( _result AND _retry_count LESS _max_retries )
    if( _working_dir )
      execute_process(
        COMMAND ${_pip_cmd}
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _pip_stdout
        ERROR_VARIABLE _pip_stderr
        WORKING_DIRECTORY ${_working_dir}
      )
    else()
      execute_process(
        COMMAND ${_pip_cmd}
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _pip_stdout
        ERROR_VARIABLE _pip_stderr
      )
    endif()

    # Print stdout always, but only print stderr on failure to avoid
    # CTest launchers interpreting pip dependency warnings as build errors
    if( _pip_stdout )
      message( STATUS "${_pip_stdout}" )
    endif()
    if( _result AND _pip_stderr )
      message( STATUS "${_pip_stderr}" )
    endif()

    if( _result )
      math( EXPR _retry_count "${_retry_count} + 1" )
      if( _retry_count LESS _max_retries )
        message( STATUS "pip install failed, retrying (${_retry_count}/${_max_retries})..." )
        execute_process( COMMAND ${CMAKE_COMMAND} -E sleep 5 )
      endif()
    endif()
  endwhile()
endif()

if( NOT _result EQUAL 0 )
  message( FATAL_ERROR "pip install exited with non-zero status: ${_result}" )
endif()
