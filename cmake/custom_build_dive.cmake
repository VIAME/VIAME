# custom_build_dive.cmake
#
# Runs DIVE's electron-builder build command and treats it as successful if the
# expected output artifact is produced, regardless of the npm exit code.
#
# Works around npm 9+/electron-builder's internal node module collector
# returning ELSPROBLEMS (missing peer dependencies for vtk.js / worker-loader)
# which propagates a non-zero exit from `npm run build:electron` even though
# the desktop build artifacts are correctly produced.
#
# Runs on every `make dive`; skips the build when no client source is newer
# than the last successful build, and installs the unpacked tree afterwards.
#
# Required variables:
#   DIVE_BUILD_CMD       - ----separated list form of the command to run
#   DIVE_NPM_INSTALL_CMD - ----separated npm ci/install command
#   DIVE_ARTIFACT        - path that must exist after the command for it to count as success
#   DIVE_SOURCE_DIR      - DIVE client source directory
#   DIVE_OUTPUT_DIR      - unpacked electron build to install
#   DIVE_INSTALL_DIR     - install destination
#   DIVE_STAMP           - file recording the last successful build

cmake_minimum_required( VERSION 3.16 )

foreach( _VAR DIVE_BUILD_CMD DIVE_NPM_INSTALL_CMD DIVE_ARTIFACT DIVE_SOURCE_DIR
    DIVE_OUTPUT_DIR DIVE_INSTALL_DIR DIVE_STAMP )
  if( NOT ${_VAR} )
    message( FATAL_ERROR "custom_build_dive.cmake requires ${_VAR}" )
  endif()
endforeach()

string( REPLACE "----" ";" _CMD "${DIVE_BUILD_CMD}" )
string( REPLACE "----" ";" _NPM_INSTALL_CMD "${DIVE_NPM_INSTALL_CMD}" )

get_filename_component( _ARTIFACT_NAME "${DIVE_ARTIFACT}" NAME )

file( GLOB_RECURSE _SOURCES LIST_DIRECTORIES false
  "${DIVE_SOURCE_DIR}/src/*"
  "${DIVE_SOURCE_DIR}/dive-common/*"
  "${DIVE_SOURCE_DIR}/platform/*"
  "${DIVE_SOURCE_DIR}/public/*"
  "${DIVE_SOURCE_DIR}/bin/*" )
file( GLOB _TOP_LEVEL LIST_DIRECTORIES false "${DIVE_SOURCE_DIR}/*" )
list( FILTER _TOP_LEVEL EXCLUDE REGEX "\\.log$" )

set( _UP_TO_DATE FALSE )
if( EXISTS "${DIVE_STAMP}" AND EXISTS "${DIVE_ARTIFACT}"
    AND EXISTS "${DIVE_INSTALL_DIR}/${_ARTIFACT_NAME}" )
  set( _UP_TO_DATE TRUE )
  foreach( _FILE ${_SOURCES} ${_TOP_LEVEL} )
    if( "${_FILE}" IS_NEWER_THAN "${DIVE_STAMP}" )
      message( STATUS "DIVE: ${_FILE} changed, rebuilding" )
      set( _UP_TO_DATE FALSE )
      break()
    endif()
  endforeach()
endif()

if( _UP_TO_DATE )
  message( STATUS "DIVE: sources unchanged, skipping build" )
  return()
endif()

if( EXISTS "${DIVE_STAMP}"
    AND "${DIVE_SOURCE_DIR}/package-lock.json" IS_NEWER_THAN "${DIVE_STAMP}" )
  message( STATUS "DIVE: package-lock.json changed, reinstalling node modules" )
  execute_process(
    COMMAND ${_NPM_INSTALL_CMD}
    WORKING_DIRECTORY "${DIVE_SOURCE_DIR}"
    RESULT_VARIABLE _NPM_RC )
  if( NOT _NPM_RC EQUAL 0 )
    message( FATAL_ERROR "DIVE: node module install failed (rc=${_NPM_RC})" )
  endif()
endif()

# A stale artifact must not make a failed rebuild look successful.
file( REMOVE "${DIVE_ARTIFACT}" )

# Capture inner command output to files instead of letting it flow to MSBuild's
# stdout/stderr. electron-builder's output contains "npm error code ELSPROBLEMS"
# lines (from its internal node-module collector) which MSBuild and the CTest
# launchers scan and count as compiler errors, marking dive's custom build
# target as failed (MSB8066 exited with code -1) even when all sub-stamps were
# written and the artifacts were produced. By capturing the output away from
# MSBuild's view, dive's target exits cleanly and downstream projects (vivia
# already succeeds; viame depends on dive at the MSBuild ProjectReference level
# and gets skipped otherwise) can build.
get_filename_component( _LOG_DIR "${DIVE_ARTIFACT}" DIRECTORY )
get_filename_component( _LOG_DIR "${_LOG_DIR}/../.." ABSOLUTE )
set( _STDOUT_FILE "${_LOG_DIR}/dive_build_stdout.log" )
set( _STDERR_FILE "${_LOG_DIR}/dive_build_stderr.log" )

execute_process(
  COMMAND ${_CMD}
  RESULT_VARIABLE _RC
  OUTPUT_FILE "${_STDOUT_FILE}"
  ERROR_FILE  "${_STDERR_FILE}"
)

if( EXISTS "${DIVE_ARTIFACT}" )
  message( STATUS "DIVE: build succeeded (artifact present, rc=${_RC})" )
  message( STATUS "DIVE: build output captured at ${_STDOUT_FILE} / ${_STDERR_FILE}" )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${DIVE_OUTPUT_DIR}" "${DIVE_INSTALL_DIR}"
    RESULT_VARIABLE _COPY_RC )
  if( NOT _COPY_RC EQUAL 0 )
    message( FATAL_ERROR "DIVE: install into ${DIVE_INSTALL_DIR} failed (rc=${_COPY_RC})" )
  endif()
  file( WRITE "${DIVE_STAMP}" "" )
else()
  # Replay captured output to stdout so a real failure isn't silent.
  if( EXISTS "${_STDOUT_FILE}" )
    file( READ "${_STDOUT_FILE}" _OUT )
    if( _OUT )
      message( STATUS "DIVE stdout:\n${_OUT}" )
    endif()
  endif()
  if( EXISTS "${_STDERR_FILE}" )
    file( READ "${_STDERR_FILE}" _ERR )
    if( _ERR )
      message( STATUS "DIVE stderr:\n${_ERR}" )
    endif()
  endif()
  message( FATAL_ERROR "DIVE: build artifact ${DIVE_ARTIFACT} was not produced (rc=${_RC})" )
endif()
