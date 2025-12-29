# Build / Install Plugin containing VXL Algorithm Implementations

# Find VXL
find_package( VXL REQUIRED )
include( ${VXL_CMAKE_DIR}/UseVXL.cmake )
include_directories( SYSTEM ${VXL_CORE_INCLUDE_DIR} )
include_directories( SYSTEM ${VXL_VCL_INCLUDE_DIR} )
include_directories( SYSTEM ${VXL_RPL_INCLUDE_DIR} )
link_directories( ${VXL_LIBRARY_DIR} )

###
# Loadable algorithms
##

set( plugin_headers
  perform_white_balancing.h
  )

set( plugin_sources
  perform_white_balancing.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_vxl_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_vxl
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( viame_vxl
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiver_algo_vxl
                       vil vil_algo
  )

set_target_properties( viame_vxl PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_vxl
  register_algorithms.cxx
  )

target_link_libraries( viame_vxl_plugin
  PUBLIC               kwiver::vital_vpm
  )

###
# Add process library
##
set( process_headers
  format_images_srm_process.h
)

set( process_sources
  register_processes.cxx
  format_images_srm_process.cxx
)

kwiver_add_plugin( viame_processes_vxl
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
                   ${private_headers}
  PRIVATE          kwiver_algo_vxl vil vil_algo
                   kwiver::sprokit_pipeline kwiver::kwiver_algo_core
                   kwiver::kwiversys kwiver::kwiver_adapter
                   kwiver::vital kwiver::vital_vpm kwiver::vital_logger
                   kwiver::vital_config kwiver::vital_util
                   kwiver::vital_exceptions
 )
