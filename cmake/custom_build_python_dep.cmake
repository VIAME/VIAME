# custom_build_python_dep.cmake
#
# This script wraps Python package builds to skip compilation when source hasn't changed.
# It checks a hash file and only runs the actual build if the hash differs.
#
# Required variables:
#   LIB_NAME       - Name of the library
#   LIB_SOURCE_DIR - Path to the library source directory
#   HASH_FILE      - Path to the hash file for comparison
#   BUILD_COMMAND  - The actual build command to run (quoted string)
#   ENV_VARS       - Environment variables (----separated KEY=VALUE pairs)
#   TMPDIR         - Temporary directory for Python builds
#   WORKING_DIR    - Working directory for the build command

cmake_minimum_required(VERSION 3.16)

# Function to get the current source hash
function(get_source_hash SOURCE_DIR OUT_HASH)
  execute_process(
    COMMAND git -C "${SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE GIT_HASH
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_RESULT
  )

  if(GIT_RESULT EQUAL 0 AND GIT_HASH)
    execute_process(
      COMMAND git -C "${SOURCE_DIR}" diff --quiet HEAD
      RESULT_VARIABLE DIFF_RESULT
    )
    if(NOT DIFF_RESULT EQUAL 0)
      set(GIT_HASH "${GIT_HASH}-dirty")
    endif()
    set(${OUT_HASH} "${GIT_HASH}" PARENT_SCOPE)
    return()
  endif()

  # Fallback: use file modification times
  set(HASH_INPUT "")
  foreach(CHECK_FILE setup.py pyproject.toml setup.cfg)
    if(EXISTS "${SOURCE_DIR}/${CHECK_FILE}")
      file(TIMESTAMP "${SOURCE_DIR}/${CHECK_FILE}" FILE_TIME)
      string(APPEND HASH_INPUT "${CHECK_FILE}:${FILE_TIME};")
    endif()
  endforeach()

  foreach(CHECK_DIR src csrc detectron2 sam2)
    if(IS_DIRECTORY "${SOURCE_DIR}/${CHECK_DIR}")
      file(GLOB_RECURSE SRC_FILES "${SOURCE_DIR}/${CHECK_DIR}/*")
      list(LENGTH SRC_FILES FILE_COUNT)
      string(APPEND HASH_INPUT "${CHECK_DIR}:${FILE_COUNT};")
    endif()
  endforeach()

  string(MD5 FALLBACK_HASH "${HASH_INPUT}")
  set(${OUT_HASH} "${FALLBACK_HASH}" PARENT_SCOPE)
endfunction()

# Validate required variables
if(NOT LIB_NAME OR NOT LIB_SOURCE_DIR OR NOT HASH_FILE OR NOT BUILD_COMMAND)
  message(FATAL_ERROR "custom_build_python_dep.cmake requires LIB_NAME, LIB_SOURCE_DIR, HASH_FILE, and BUILD_COMMAND")
endif()

# Get current source hash
get_source_hash("${LIB_SOURCE_DIR}" CURRENT_HASH)

# Read stored hash if it exists
set(STORED_HASH "")
if(EXISTS "${HASH_FILE}")
  file(READ "${HASH_FILE}" STORED_HASH)
  string(STRIP "${STORED_HASH}" STORED_HASH)
endif()

# Compare hashes
if(CURRENT_HASH STREQUAL STORED_HASH)
  message(STATUS "${LIB_NAME}: Source unchanged (${CURRENT_HASH}), skipping Python build")
else()
  if(STORED_HASH)
    message(STATUS "${LIB_NAME}: Source changed (${STORED_HASH} -> ${CURRENT_HASH}), running Python build")
  else()
    message(STATUS "${LIB_NAME}: First build, running Python build")
  endif()

  # Convert ----separated env vars back to list
  if(ENV_VARS)
    string(REPLACE "----" ";" ENV_VARS_LIST "${ENV_VARS}")
  else()
    set(ENV_VARS_LIST "")
  endif()

  # Add TMPDIR to environment
  if(TMPDIR)
    list(APPEND ENV_VARS_LIST "TMPDIR=${TMPDIR}")
  endif()

  # BUILD_COMMAND comes in as a semicolon-separated list (CMake list format)
  # Just use it directly as a list
  set(BUILD_ARGS "${BUILD_COMMAND}")

  # Execute the build command
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ${ENV_VARS_LIST} ${BUILD_ARGS}
    WORKING_DIRECTORY "${WORKING_DIR}"
    RESULT_VARIABLE BUILD_RESULT
    COMMAND_ECHO STDOUT
  )

  if(NOT BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "${LIB_NAME}: Build failed with exit code ${BUILD_RESULT}")
  endif()

  # Store the new hash only after successful build
  file(WRITE "${HASH_FILE}" "${CURRENT_HASH}")
endif()
