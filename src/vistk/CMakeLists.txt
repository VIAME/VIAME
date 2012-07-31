include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fvisibility=hidden __VISTK_HAVE_GCC_VISIBILITY)
vistk_configure_file(config.h
  "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h"
  __VISTK_HAVE_GCC_VISIBILITY)

set(VISTK_BUILT_FROM_GIT)

if (GIT_FOUND)
  set(configure_code "
if (IS_DIRECTORY \"${vistk_source_dir}/.git\")
  set(VISTK_BUILT_FROM_GIT TRUE)

  execute_process(
    COMMAND           \"${GIT_EXECUTABLE}\"
                      rev-parse
                      HEAD
    WORKING_DIRECTORY \"${vistk_source_dir}\"
    RESULT_VARIABLE   git_return
    OUTPUT_VARIABLE   vistk_git_hash)
  execute_process(
    COMMAND           \"${GIT_EXECUTABLE}\"
                      rev-parse
                      --short
                      HEAD
    WORKING_DIRECTORY \"${vistk_source_dir}\"
    RESULT_VARIABLE   git_return
    OUTPUT_VARIABLE   vistk_git_hash_short)
  execute_process(
    COMMAND           \"${GIT_EXECUTABLE}\"
                      diff
                      --no-ext-diff
                      --quiet
                      --exit-code
    WORKING_DIRECTORY \"${vistk_source_dir}\"
    RESULT_VARIABLE   git_return)

  string(STRIP \"\${vistk_git_hash}\" vistk_git_hash)
  string(STRIP \"\${vistk_git_hash_short}\" vistk_git_hash_short)

  if (git_return)
    set(vistk_git_dirty \"dirty\")
  endif ()
endif ()
")
else ()
  if ("$Format:$" STREQUAL "")
    set(vistk_git_hash       "$Format:%H$")
    set(vistk_git_hash_short "$Format:%h$")

    option(VISTK_IS_PATCHED "Set to ON if patches are applied on top of a released tarball" OFF)
    if (VISTK_IS_PATCHED)
      set(vistk_git_dirty "dirty")
    endif ()
  endif ()
endif ()

vistk_configure_file_always(version.h
  "${CMAKE_CURRENT_SOURCE_DIR}/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/version.h"
  vistk_version_major
  vistk_version_minor
  vistk_version_patch
  vistk_version
  VISTK_BUILT_FROM_GIT
  vistk_git_hash
  vistk_git_hash_short
  vistk_git_dirty)

add_subdirectory(pipeline)
add_subdirectory(utilities)
add_subdirectory(pipeline_util)
add_subdirectory(image)
add_subdirectory(scoring)

if (VISTK_ENABLE_PYTHON)
  add_subdirectory(python)
endif ()

install(
  FILES       "${CMAKE_CURRENT_BINARY_DIR}/config.h"
              "${CMAKE_CURRENT_BINARY_DIR}/version.h"
  DESTINATION include/vistk
  COMPONENT   development)
