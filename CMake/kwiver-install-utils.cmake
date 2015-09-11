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
        "${utils_dir}/FindPROJ.cmake"
        "${utils_dir}/FindEigen3.cmake"
  DESTINATION "${kwiver_cmake_install_dir}"
  )

install(
  DIRECTORY "${utils_dir}/utils"
            "${utils_dir}/tools"
  DESTINATION "${kwiver_cmake_install_dir}"
  )
