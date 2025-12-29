# Build/install plugin for burnout CNN support

include_directories( SYSTEM ${VIDTK_INCLUDE_DIRS} )

set( headers
  burnout_detector.h
  burnout_image_enhancer.h
  burnout_pixel_classification.h
  burnout_track_descriptors.h
  )

set( sources
  burnout_detector.cxx
  burnout_image_enhancer.cxx
  burnout_pixel_classification.cxx
  burnout_track_descriptors.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_burnout_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_burnout
  ${headers}
  ${sources}
  )

find_package( Boost ${KWIVER_BOOST_VERSION} REQUIRED
  COMPONENTS
    filesystem
    system )

include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
link_directories( ${Boost_LIBRARY_DIRS} )

target_link_libraries( viame_burnout
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
  PRIVATE              vidtk_descriptor
                       vidtk_utilities_no_process
                       vidtk_pipelines
                       kwiver::kwiversys
                       ${Boost_SYSTEM_LIBRARY}
                       ${Boost_FILESYSTEM_LIBRARY}
                       kwiver::kwiver_algo_ocv
                       kwiver::kwiver_algo_vxl
  )

set_target_properties( viame_burnout PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_burnout
  register_algorithms.cxx
  )

target_link_libraries( viame_burnout_plugin
  PUBLIC               kwiver::vital_vpm
  )
