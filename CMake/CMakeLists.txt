#
# CMake rules for installing common CMake support files and functions
#

kwiver_install(
  FILES kwiver-flags.cmake kwiver-utils.cmake
  DESTINATION "${kwiver_cmake_install_dir}"
  )

kwiver_install(
  DIRECTORY utils
  DESTINATION "${kwiver_cmake_install_dir}"
  )
