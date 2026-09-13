# The CMake files an out-of-tree plugin gets
#
# `viame-config.cmake` puts this directory on the consumer's module path, so
# what lands here is VIAME's public CMake surface.
#
# Kwiver's `kwiver-install-utils.cmake` installed thirteen files and five
# directories: its flags files, its config checks, its python setup, its
# doxygen and sphinx helpers, the `future` back-port directory that has only
# a README in it. None of that is usable from outside the build it was
# written for -- the flags files append to a global property nothing outside
# reads, the config checks want probe sources that were not installed -- and
# `examples/plugin_creation`, which is the documented way to write a plugin
# against an installed VIAME, uses none of it: it calls `add_library` and
# `target_link_libraries( kwiver::vital )` and nothing else.
#
# What is installed is what a plugin could actually call.

install(
  FILES "${CMAKE_CURRENT_LIST_DIR}/viame-targets.cmake"
        "${CMAKE_CURRENT_LIST_DIR}/viame-tests.cmake"
        "${CMAKE_CURRENT_LIST_DIR}/viame-test-setup.cmake"
        "${CMAKE_CURRENT_LIST_DIR}/viame-flags-check.cmake"
  DESTINATION "${viame_cmake_install_dir}"
  )

if( VIAME_ENABLE_PYTHON )
  install(
    FILES "${CMAKE_CURRENT_LIST_DIR}/viame-python.cmake"
          "${CMAKE_CURRENT_LIST_DIR}/viame-setup-python.cmake"
    DESTINATION "${viame_cmake_install_dir}"
    )
endif()
