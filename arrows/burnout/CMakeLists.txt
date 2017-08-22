# Build/install plugin for burnout CNN support
#

set( headers
  burnout_track_descriptors.h
)

set( sources
  burnout_track_descriptors.cxx
  )

set( burnout_linked_libs
  ${VIDTK_LIBRARIES}
  kwiversys
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  kwiver_algo_ocv
  kwiver_algo_vxl )

include_directories( SYSTEM ${VIDTK_INCLUDE_DIRS} )

include_directories( SYSTEM ${VIDTK_INCLUDE_DIRS} )

kwiver_install_headers(
  SUBDIR     arrows/burnout
  ${headers}
  )
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_burnout_export.h
  NOPATH   SUBDIR     arrows/burnout
  )

kwiver_add_library( kwiver_algo_burnout
  ${headers}
  ${sources}
  )

target_link_libraries( kwiver_algo_burnout
  PUBLIC               vital_algo
  PRIVATE              ${burnout_linked_libs}
                       vil vidtk_descriptor
  )

algorithms_create_plugin( kwiver_algo_burnout
  register_algorithms.cxx
  )
