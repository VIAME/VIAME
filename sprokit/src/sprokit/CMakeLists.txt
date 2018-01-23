#
#
#

set(SPROKIT_BUILT_FROM_GIT)

if ("$Format:$" STREQUAL "")
  set(sprokit_git_hash       "$Format:%H$")
  set(sprokit_git_hash_short "$Format:%h$")

  option(SPROKIT_IS_PATCHED "Set to ON if patches are applied on top of a released tarball" OFF)
  if (SPROKIT_IS_PATCHED)
    set(sprokit_git_dirty "dirty")
  endif ()
elseif (GIT_FOUND)
  set(configure_code "
if (IS_DIRECTORY \"${sprokit_source_dir}/.git\")
  set(SPROKIT_BUILT_FROM_GIT TRUE)

  execute_process(
    COMMAND           \"${GIT_EXECUTABLE}\"
                      rev-parse
                      HEAD
    WORKING_DIRECTORY \"${sprokit_source_dir}\"
    RESULT_VARIABLE   git_return
    OUTPUT_VARIABLE   sprokit_git_hash)
  execute_process(
    COMMAND           \"${GIT_EXECUTABLE}\"
                      rev-parse
                      --short
                      HEAD
    WORKING_DIRECTORY \"${sprokit_source_dir}\"
    RESULT_VARIABLE   git_return
    OUTPUT_VARIABLE   sprokit_git_hash_short)
  execute_process(
    COMMAND           \"${GIT_EXECUTABLE}\"
                      diff
                      --no-ext-diff
                      --quiet
                      --exit-code
    WORKING_DIRECTORY \"${sprokit_source_dir}\"
    RESULT_VARIABLE   git_return)

  string(STRIP \"\${sprokit_git_hash}\" sprokit_git_hash)
  string(STRIP \"\${sprokit_git_hash_short}\" sprokit_git_hash_short)

  if (git_return)
    set(sprokit_git_dirty \"dirty\")
  endif ()

  message(STATUS \"version: \${sprokit_version}\")
  message(STATUS \"git hash: \${sprokit_git_hash}\")
  message(STATUS \"git short hash: \${sprokit_git_hash_short}\")
  message(STATUS \"git dirty: \${sprokit_git_dirty}\")
endif ()
")
else ()
  set(sprokit_git_hash       "<unknown>")
  set(sprokit_git_hash_short "<unknown>")
  set(sprokit_git_dirty      "<unknown>")
endif ()

sprokit_configure_file_always(version.h
  "${CMAKE_CURRENT_SOURCE_DIR}/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/version.h"
  KWIVER_VERSION_MAJOR
  KWIVER_VERSION_MINOR
  KWIVER_VERSION_PATCH
  KWIVER_VERSION
  SPROKIT_BUILT_FROM_GIT
  sprokit_git_hash
  sprokit_git_hash_short
  sprokit_git_dirty)

set(configure_code)

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory(python)
endif ()

add_subdirectory(pipeline)
add_subdirectory(pipeline_util)
add_subdirectory(scoring)
add_subdirectory(tools)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h"
  DESTINATION include/sprokit
  COMPONENT   development)
