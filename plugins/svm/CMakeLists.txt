# Build/install plugin for SVM support

find_package( OpenCV REQUIRED )

# Find libsvm in the install prefix (lib on Linux, bin or lib on Windows)
find_library( SVM_LIBRARY
  NAMES svm libsvm svm.2 libsvm.so.2
  PATHS ${VIAME_INSTALL_PREFIX}/lib ${VIAME_INSTALL_PREFIX}/bin
  NO_DEFAULT_PATH
)

if( NOT SVM_LIBRARY )
  message( FATAL_ERROR "Could not find libsvm library in ${VIAME_INSTALL_PREFIX}" )
endif()

set( headers
  refine_detections_svm.h
  train_detector_svm.h )

set( sources
  refine_detections_svm.cxx
  train_detector_svm.cxx )

kwiver_install_headers(
  SUBDIR     viame
  ${headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_svm_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_svm
  ${headers}
  ${sources}
  )

target_include_directories( viame_svm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

find_package( Boost ${KWIVER_BOOST_VERSION} REQUIRED
  COMPONENTS
    filesystem
    regex
    system )

include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
link_directories( ${Boost_LIBRARY_DIRS} )

target_link_libraries( viame_svm
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::kwiver_algo_core
                       ${OpenCV_LIBS}
  PRIVATE              kwiver::vital_config
                       kwiver::kwiversys
                       ${Boost_SYSTEM_LIBRARY}
                       ${Boost_FILESYSTEM_LIBRARY}
                       ${Boost_REGEX_LIBRARY}
                       ${SVM_LIBRARY}
  )

set_target_properties( viame_svm PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_svm
  register_algorithms.cxx
  )

target_link_libraries( viame_svm_plugin
  PUBLIC               kwiver::vital_vpm
  )

# Add process library
set( process_headers
  process_query_process.h
  train_svm_models_process.h )

set( process_sources
  register_processes.cxx
  process_query_process.cxx
  train_svm_models_process.cxx
)

kwiver_add_plugin( viame_processes_svm
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
  PRIVATE          kwiver::vital
                   kwiver::vital_algo
                   kwiver::vital_config
                   kwiver::vital_logger
                   kwiver::vital_util
                   kwiver::vital_vpm
                   kwiver::sprokit_pipeline
                   ${SVM_LIBRARY}
 )

target_include_directories( viame_processes_svm PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/plugins>
  )

# Optionally link CppDB for database backend support in process_query
if( VIAME_ENABLE_POSTGRESQL )
  target_compile_definitions( viame_processes_svm PRIVATE VIAME_ENABLE_CPPDB )
  target_link_libraries( viame_processes_svm PRIVATE viame_cppdb )
endif()
