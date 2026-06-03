# dive_build_wrapper.cmake
#
# Runs DIVE's electron-builder build command and treats it as successful if the
# expected output artifact is produced, regardless of the npm exit code.
#
# Works around npm 9+/electron-builder's internal node module collector
# returning ELSPROBLEMS (missing peer dependencies for vtk.js / worker-loader)
# which propagates a non-zero exit from `npm run build:electron` even though
# the desktop build artifacts are correctly produced.
#
# Required variables:
#   DIVE_BUILD_CMD - ----separated list form of the command to run
#   DIVE_ARTIFACT  - path that must exist after the command for it to count as success

cmake_minimum_required( VERSION 3.16 )

if( NOT DIVE_BUILD_CMD OR NOT DIVE_ARTIFACT )
  message( FATAL_ERROR "dive_build_wrapper.cmake requires DIVE_BUILD_CMD and DIVE_ARTIFACT" )
endif()

string( REPLACE "----" ";" _CMD "${DIVE_BUILD_CMD}" )

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
