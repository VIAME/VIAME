# Build / Install Plugin containing SVM algorithm implementations

set( plugin_headers
  refine_detections_with_svm.h
  )

kwiver_install_headers(
  SUBDIR     arrows/svm
  ${plugin_headers}
  )

kwiver_install_headers(
        ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_svm_export.h
        NOPATH   SUBDIR     arrows/svm
)

set( plugin_sources
  refine_detections_with_svm.cxx
  )

kwiver_add_library( kwiver_algo_svm
  ${plugin_headers}
  ${plugin_sources}
  )

if( WIN32 )
  set( SVM_LIBRARY libsvm )
else()
  set( SVM_LIBRARY svm )
endif()

target_link_libraries( kwiver_algo_svm
  PUBLIC               kwiver_algo_core
                       ${OpenCV_LIBS}
  PRIVATE              vital_config
                       kwiversys
                       ${Boost_SYSTEM_LIBRARY}
                       ${Boost_FILESYSTEM_LIBRARY}
                       ${Boost_REGEX_LIBRARY}
                       ${SVM_LIBRARY}
  )

algorithms_create_plugin( kwiver_algo_svm
  register_algorithms.cxx
  )

