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

viame_add_library( vital_applets
  ${public_headers}
  ${source_files}
)

target_link_libraries( vital_applets
  PUBLIC      vital_util
              vital_config
              vital_logger
              vital_vpm  # uses pluggable
  )

###
# Install header files

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_applets_export.h
  NOPATH
  SUBDIR      viame/algorithm_framework/applets
  )

# Republished under the prefix the imported sources use.
viame_lite_generated( vital_applets_export.h algorithm_framework/applets )
