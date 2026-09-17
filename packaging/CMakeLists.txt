###
# Packaging metadata
#
# `setup.py egg_info` writes the `viame.egg-info` that declares VIAME's
# python entry points, and `viame.plugins.discovery` reads those groups.
# Nothing else declares them: the `say` and `they_say` implementations in
# `viame.test_interface` register through `viame.python_plugins` and vanish
# from the registry without this, taking the only in-tree exercise of the
# entry point discovery path with them.
#
# This was `python/CMakeLists.txt`'s `install_egg_info` target until
# P11-T02b deleted that directory along with the rest of `python/`.
##
if( NOT VIAME_ENABLE_PYTHON )
  return()
endif()

set( _viame_egg_base "${KWIVER_BINARY_DIR}/${python_site_packages}" )

add_custom_target( install_egg_info ALL
  COMMENT "Writing the python entry point metadata"
  # `setup.py egg_info` refuses an `--egg-base` that does not exist, and this
  # target has no dependency that would have created it. A warm tree hides
  # that: something else made the directory on an earlier run.
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_viame_egg_base}"
  COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
          egg_info --egg-base "${_viame_egg_base}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )

install(
  DIRECTORY   "${_viame_egg_base}/viame.egg-info"
  DESTINATION "${python_site_packages}/"
  )
