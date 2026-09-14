###
# Things the rest of VIAME uses and nothing owns
#
# The file, path and image handling, the track geometry and the
# classification averaging, the atomic write,
# the applet base class that runs an installed python tool, and the
# pipeline-file rewriting the training tools do. None of it is an algorithm
# or a process: nothing here registers, and `viame_utilities` has no
# `register.cxx` for that reason.
#
# It was `plugins/core`; lite-library-layout.md §3 names the six pieces.
##

set( plugin_headers
  applet_attributes.h
  atomic_output.h
  manipulate_pipelines.h
  python_script_applet.h
  utilities_file.h
  utilities_image.h
  utilities_target_clfr.h
  utilities_tracks.h
  )

set( plugin_sources
  manipulate_pipelines.cxx
  python_script_applet.cxx
  utilities_file.cxx
  utilities_image.cxx
  utilities_target_clfr.cxx
  utilities_tracks.cxx
  )

viame_install_headers(
  SUBDIR     viame/utilities
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_utilities_export.h
  NOPATH   SUBDIR     viame/utilities
  )

viame_add_library( viame_utilities
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_utilities PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_utilities
  PUBLIC               kwiver::vital
                       kwiver::vital_applets
                       kwiver::vital_config
                       kwiver::vital_logger
                       kwiver::vital_util
  )

# miniz is the NPZ writer `utilities_file` uses. Static and vendored, so
# `$<BUILD_INTERFACE:...>` as well as PRIVATE: a static library linked even
# privately is recorded in the target's interface, and `install( EXPORT )`
# then refuses it for not being in the export set.
target_link_libraries( viame_utilities
  PRIVATE              $<BUILD_INTERFACE:viame::miniz>
  )

# The guards around the NPZ code are vestigial -- zlib was found or not
# found, miniz always is -- and left alone rather than edited out of working
# code.
target_compile_definitions( viame_utilities PUBLIC VIAME_ENABLE_ZLIB )

set_target_properties( viame_utilities PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

###
# Python
#
# `vital_registration` is the base class every python algorithm in VIAME
# derives from; the rest is imported by other packages.
##
if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( utilities )
endif()
