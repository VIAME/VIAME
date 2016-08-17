###
# CMake file for matlab bindings

set( sources
  matlab_array.cxx
  matlab_exception.cxx
  matlab_engine.cxx
  matlab_image_object_detector.cxx
  )

set( public_headers
  matlab_array.h
  matlab_exception.h
  matlab_engine.h
  matlab_image_object_detector.h
  )

set( private_headers

  )

include_directories( "${Matlab_INCLUDE_DIRS}" )

kwiver_install_headers(
  ${public_headers}
  SUBDIR     vital/bindings/matlab
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_matlab_export.h
  SUBDIR     vital/bindings/matlab
  NOPATH
  )

kwiver_add_library( vital_matlab
  ${public_headers}
  ${private_headers}
  ${sources}
  )

target_link_libraries( vital_matlab
  PRIVATE     vital
              ${Matlab_LIBRARIES}
              ${OpenCV_LIBS}
              kwiver_algo_ocv
  )
