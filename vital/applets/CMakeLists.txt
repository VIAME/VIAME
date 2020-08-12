#
# CMake file for the vital tools
#

set( sources
  register_tools.cxx

  config_explorer.cxx
  )

set( headers
  config_explorer.h
  cxxopts.hpp
  )


###
# Add our plugin path to the default list
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

###
# tools applet support library
kwiver_add_library( kwiver_tools_applet
               kwiver_applet.h
               kwiver_applet.cxx
)

target_link_libraries( kwiver_tools_applet
  PUBLIC      vital_util
              ${VITAL_BOOST_REGEX}
  )


###
# Add applet for tools plugin
kwiver_add_plugin( vital_applets
  SUBDIR       ${kwiver_plugin_applets_subdir}
  SOURCES      ${sources} ${headers}
  PRIVATE      vital
               vital_vpm
               vital_klv
               vital_algo
               vital_logger
               kwiversys
               kwiver_tools_applet
)

###
# Install header files
kwiver_install_headers(
  kwiver_applet.h
  cxxopts.hpp
  SUBDIR      vital/applets
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_tools_applet_export.h
  NOPATH
  SUBDIR      vital/applets
  )
