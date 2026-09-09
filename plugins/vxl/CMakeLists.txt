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

# enhance_images moved to plugins/opencv, which had a byte identical twin of
# it; perform_white_balancing was removed as unused. Only the srm process is
# left here, and P3-T05 takes that off VXL too.
set( plugin_headers )

set( plugin_sources )

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

target_include_directories( viame_processes_vxl PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
