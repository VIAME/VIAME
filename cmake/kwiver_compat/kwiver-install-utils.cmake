# Installation logic for the CMake utilities an out-of-tree plugin needs
#
# The list is what `viame-config.cmake` puts on a consumer's module path.
# P5-T06 took the finders for dependencies that are gone -- FFmpeg, log4cxx,
# PROJ -- out of it.
#
# Variables that modify function:
#
#   kwiver_cmake_install_dir
#     - Directory to install files to
#
set(utils_dir "${CMAKE_CURRENT_LIST_DIR}")

if(NOT SKBUILD)
  install(
    FILES "${utils_dir}/kwiver-utils.cmake"
          "${utils_dir}/kwiver-flags.cmake"
          "${utils_dir}/kwiver-utils.cmake"
          "${utils_dir}/kwiver-configcheck.cmake"
          "${utils_dir}/kwiver-flags-gnu.cmake"
          "${utils_dir}/kwiver-flags-msvc.cmake"
          "${utils_dir}/kwiver-flags-clang.cmake"
          "${utils_dir}/kwiver-configcheck.cmake"
          "${utils_dir}/kwiver-cmake-future.cmake"
          "${utils_dir}/kwiver-setup-python.cmake"
          "${utils_dir}/CommonFindMacros.cmake"
          "${utils_dir}/FindSphinx.cmake"
    DESTINATION "${kwiver_cmake_install_dir}"
    )

  install(
    DIRECTORY "${utils_dir}/utils"
              "${utils_dir}/tools"
              "${utils_dir}/configcheck"
              "${utils_dir}/templates"
              "${utils_dir}/future"
    DESTINATION "${kwiver_cmake_install_dir}"
    )
endif()
