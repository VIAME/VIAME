# Installation logic for kwiver CMake utilities
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
          "${utils_dir}/FindEigen3.cmake"
          "${utils_dir}/FindFFMPEG.cmake"
          "${utils_dir}/FindLog4cxx.cmake"
          "${utils_dir}/FindPROJ.cmake"
          "${utils_dir}/FindSphinx.cmake"
          "${utils_dir}/FindTinyXML.cmake"
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
