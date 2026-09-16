#
# CMake file for the vital tools
#

set(public_headers
  applet_context.h
  kwiver_applet.h
  )

set(source_files
  kwiver_applet.cxx
  )

###
# tools applet support library

viame_add_library( viame_applets
  ${public_headers}
  ${source_files}
)

target_link_libraries( viame_applets
  PUBLIC      viame_util
              viame_config
              viame_logger
              viame_plugin  # uses pluggable
  )

###
# Install header files

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_applets_export.h
  NOPATH
  SUBDIR      viame/algorithm_framework/applets
  )

# Republished under the prefix the imported sources use.
viame_lite_generated( viame_applets_export.h algorithm_framework/applets )
