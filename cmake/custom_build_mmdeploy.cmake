# custom_build_mmdeploy.cmake
#
# This script wraps mmdeploy builds to skip compilation when source hasn't changed.
# mmdeploy has both C++ and Python components that need conditional building.
#
# Required variables:
#   LIB_NAME        - Name of the library
#   LIB_SOURCE_DIR  - Path to the library source directory
#   HASH_FILE       - Path to the hash file for comparison
#   CPP_BUILD_CMD   - The C++ build command
#   CPP_INSTALL_CMD - The C++ install command
#   PYTHON_BUILD_CMD - The Python build command
#   ENV_VARS        - Environment variables (----separated KEY=VALUE pairs)
#   TMPDIR          - Temporary directory for Python builds
#   WORKING_DIR     - Working directory for the build command

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
  foreach(CHECK_FILE setup.py pyproject.toml setup.cfg CMakeLists.txt)
    if(EXISTS "${SOURCE_DIR}/${CHECK_FILE}")
      file(TIMESTAMP "${SOURCE_DIR}/${CHECK_FILE}" FILE_TIME)
      string(APPEND HASH_INPUT "${CHECK_FILE}:${FILE_TIME};")
    endif()
  endforeach()

  foreach(CHECK_DIR src csrc mmdeploy)
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
if(NOT LIB_NAME OR NOT LIB_SOURCE_DIR OR NOT HASH_FILE)
  message(FATAL_ERROR "custom_build_mmdeploy.cmake requires LIB_NAME, LIB_SOURCE_DIR, and HASH_FILE")
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
  message(STATUS "${LIB_NAME}: Source unchanged (${CURRENT_HASH}), skipping C++ and Python builds")
else()
  if(STORED_HASH)
    message(STATUS "${LIB_NAME}: Source changed (${STORED_HASH} -> ${CURRENT_HASH}), running full build")
  else()
    message(STATUS "${LIB_NAME}: First build, running full build")
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

  # Run C++ build - command comes as semicolon-separated list
  if(CPP_BUILD_CMD)
    message(STATUS "${LIB_NAME}: Running C++ build...")
    set(CPP_BUILD_ARGS "${CPP_BUILD_CMD}")
    execute_process(
      COMMAND ${CPP_BUILD_ARGS}
      WORKING_DIRECTORY "${WORKING_DIR}"
      RESULT_VARIABLE BUILD_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "${LIB_NAME}: C++ build failed with exit code ${BUILD_RESULT}")
    endif()
  endif()

  # Run C++ install
  if(CPP_INSTALL_CMD)
    message(STATUS "${LIB_NAME}: Running C++ install...")
    set(CPP_INSTALL_ARGS "${CPP_INSTALL_CMD}")
    execute_process(
      COMMAND ${CPP_INSTALL_ARGS}
      WORKING_DIRECTORY "${WORKING_DIR}"
      RESULT_VARIABLE INSTALL_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT INSTALL_RESULT EQUAL 0)
      message(FATAL_ERROR "${LIB_NAME}: C++ install failed with exit code ${INSTALL_RESULT}")
    endif()
  endif()

  # Run Python build
  if(PYTHON_BUILD_CMD)
    message(STATUS "${LIB_NAME}: Running Python build...")
    set(PYTHON_BUILD_ARGS "${PYTHON_BUILD_CMD}")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env ${ENV_VARS_LIST} ${PYTHON_BUILD_ARGS}
      WORKING_DIRECTORY "${WORKING_DIR}"
      RESULT_VARIABLE PY_BUILD_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT PY_BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "${LIB_NAME}: Python build failed with exit code ${PY_BUILD_RESULT}")
    endif()
  endif()

  # Store the new hash only after successful build
  file(WRITE "${HASH_FILE}" "${CURRENT_HASH}")
endif()
