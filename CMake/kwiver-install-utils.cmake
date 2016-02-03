# Installation logic for MAP-Tk CMake utilities
#
# Variables that modify function:
#
#   kwiver_cmake_install_dir
#     - Directory to install files to
#
set(utils_dir "${CMAKE_CURRENT_LIST_DIR}")

install(
  FILES "${utils_dir}/kwiver-utils.cmake"
        "${utils_dir}/FindEigen3.cmake"
        "${utils_dir}/FindLog4cxx.cmake"
        "${utils_dir}/vital-flags.cmake"
        "${utils_dir}/vital-flags-gnu.cmake"
        "${utils_dir}/vital-flags-msvc.cmake"
        "${utils_dir}/vital-flags-clang.cmake"
        "${utils_dir}/kwiver-configcheck.cmake"
  DESTINATION "${kwiver_cmake_install_dir}"
  )

install(
  DIRECTORY "${utils_dir}/utils"
            "${utils_dir}/tools"
            "${utils_dir}/configcheck"
  DESTINATION "${kwiver_cmake_install_dir}"
  )
