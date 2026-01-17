# custom_pip_check_install.cmake
#
# This script wraps pip install to skip installation when the input hash
# hasn't changed. It uses a hash file to track the installed state.
#
# Required variables:
#   HASH_INPUT       - String to hash for comparison (e.g., deps list or package version)
#   HASH_FILE        - Path to the hash file for comparison
#   PIP_INSTALL_CMD  - Full pip install command (----separated)
#
# Optional variables:
#   PKG_NAME         - Package/group name for display (default: "python-deps")
#   ENV_VARS         - Environment variables (----separated KEY=VALUE pairs)

cmake_minimum_required( VERSION 3.16 )

# Validate required variables
if( NOT HASH_INPUT OR NOT HASH_FILE OR NOT PIP_INSTALL_CMD )
  message( FATAL_ERROR "custom_pip_check_install.cmake requires HASH_INPUT, HASH_FILE, and PIP_INSTALL_CMD" )
endif()

# Default package name for display
if( NOT PKG_NAME )
  set( PKG_NAME "python-deps" )
endif()

# Compute hash of the input
string( MD5 CURRENT_HASH "${HASH_INPUT}" )

# Read stored hash if it exists
set( STORED_HASH "" )
if( EXISTS "${HASH_FILE}" )
  file( READ "${HASH_FILE}" STORED_HASH )
  string( STRIP "${STORED_HASH}" STORED_HASH )
endif()

# Compare hashes
if( CURRENT_HASH STREQUAL STORED_HASH )
  message( STATUS "${PKG_NAME}: Unchanged (${HASH_INPUT}), skipping pip install" )
  return()
endif()

if( STORED_HASH )
  message( STATUS "${PKG_NAME}: Changed, running pip install (${HASH_INPUT})" )
else()
  message( STATUS "${PKG_NAME}: First install, running pip install (${HASH_INPUT})" )
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
message( STATUS "${PKG_NAME}: Running pip install..." )
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
  message( FATAL_ERROR "${PKG_NAME}: pip install failed with exit code ${PIP_RESULT}" )
endif()

# Store the hash only after successful install
file( WRITE "${HASH_FILE}" "${CURRENT_HASH}" )
message( STATUS "${PKG_NAME}: Successfully installed, hash saved" )
