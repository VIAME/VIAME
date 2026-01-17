# custom_pip_check_install.cmake
#
# This script wraps pip install to skip installation when the dependency list
# hasn't changed. It uses a hash file to track the installed state.
#
# Required variables:
#   DEPS_HASH_INPUT  - String to hash for comparison (e.g., comma-separated deps list)
#   HASH_FILE        - Path to the hash file for comparison
#   PIP_INSTALL_CMD  - Full pip install command (----separated)
#
# Optional variables:
#   ENV_VARS         - Environment variables (----separated KEY=VALUE pairs)

cmake_minimum_required( VERSION 3.16 )

# Validate required variables
if( NOT DEPS_HASH_INPUT OR NOT HASH_FILE OR NOT PIP_INSTALL_CMD )
  message( FATAL_ERROR "custom_pip_check_install.cmake requires DEPS_HASH_INPUT, HASH_FILE, and PIP_INSTALL_CMD" )
endif()

# Compute hash of the dependency list
string( MD5 CURRENT_HASH "${DEPS_HASH_INPUT}" )

# Read stored hash if it exists
set( STORED_HASH "" )
if( EXISTS "${HASH_FILE}" )
  file( READ "${HASH_FILE}" STORED_HASH )
  string( STRIP "${STORED_HASH}" STORED_HASH )
endif()

# Compare hashes
if( CURRENT_HASH STREQUAL STORED_HASH )
  message( STATUS "python-deps: Dependencies unchanged, skipping pip install" )
  return()
endif()

if( STORED_HASH )
  message( STATUS "python-deps: Dependencies changed, running pip install" )
else()
  message( STATUS "python-deps: First install, running pip install" )
endif()

# Convert ----separated pip command back to list
string( REPLACE "----" ";" PIP_INSTALL_CMD_LIST "${PIP_INSTALL_CMD}" )

# Convert ----separated env vars back to list
if( ENV_VARS )
  string( REPLACE "----" ";" ENV_VARS_LIST "${ENV_VARS}" )
  # Convert <PS> path separator to platform-specific separator in each env var
  set( PROCESSED_ENV_VARS )
  foreach( ENV_VAR IN LISTS ENV_VARS_LIST )
    if( WIN32 )
      # On Windows, <PS> should become ; but we need to escape it with \; for CMake
      string( REPLACE "<PS>" "\\;" ENV_VAR "${ENV_VAR}" )
    else()
      # On Unix, <PS> should become :
      string( REPLACE "<PS>" ":" ENV_VAR "${ENV_VAR}" )
    endif()
    list( APPEND PROCESSED_ENV_VARS "${ENV_VAR}" )
  endforeach()
  set( ENV_VARS_LIST "${PROCESSED_ENV_VARS}" )
else()
  set( ENV_VARS_LIST "" )
endif()

# Run pip install
message( STATUS "python-deps: Running pip install..." )
if( ENV_VARS_LIST )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ${ENV_VARS_LIST} ${PIP_INSTALL_CMD_LIST}
    RESULT_VARIABLE PIP_RESULT
    COMMAND_ECHO STDOUT
  )
else()
  execute_process(
    COMMAND ${PIP_INSTALL_CMD_LIST}
    RESULT_VARIABLE PIP_RESULT
    COMMAND_ECHO STDOUT
  )
endif()

if( NOT PIP_RESULT EQUAL 0 )
  message( FATAL_ERROR "python-deps: pip install failed with exit code ${PIP_RESULT}" )
endif()

# Store the hash only after successful install
file( WRITE "${HASH_FILE}" "${CURRENT_HASH}" )
message( STATUS "python-deps: Successfully installed, hash saved" )
