###
# CMake file for matlab bindings

set( sources
  matlab_util.cxx
  matlab_exception.cxx
  matlab_engine.cxx
  matlab_image_object_detector.cxx
  matlab_image_filter.cxx
  matlab_detection_output.cxx
  )

set( public_headers
  mxtypes.h
  mxarray.h
  matlab_util.h
  matlab_exception.h
  matlab_engine.h
  matlab_image_object_detector.h
  matlab_image_filter.h
  matlab_detection_output.h
)

set( private_headers
  )

include_directories( "${Matlab_INCLUDE_DIRS}" )

kwiver_install_headers(
  ${public_headers}
  SUBDIR     arrows/matlab
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_matlab_export.h
  SUBDIR     arrows/matlab
  NOPATH
  )

kwiver_add_library( kwiver_algo_matlab
  ${public_headers}
  ${private_headers}
  ${sources}
  )

target_link_libraries( kwiver_algo_matlab
  PUBLIC               vital_algo
                       ${Matlab_LIBRARIES}
                       kwiver_algo_ocv
                       kwiversys
  )

algorithms_create_plugin( kwiver_algo_matlab
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
