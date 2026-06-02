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

execute_process(
  COMMAND ${_CMD}
  RESULT_VARIABLE _RC
  COMMAND_ECHO STDOUT
)

if( EXISTS "${DIVE_ARTIFACT}" )
  if( NOT _RC EQUAL 0 )
    message( STATUS "DIVE: build command returned ${_RC}, but ${DIVE_ARTIFACT} was produced - treating as success" )
  endif()
else()
  message( FATAL_ERROR "DIVE: build artifact ${DIVE_ARTIFACT} was not produced (rc=${_RC})" )
endif()
