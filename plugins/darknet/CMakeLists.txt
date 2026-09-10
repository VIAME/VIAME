# Build/install plugin for darknet CNN support

# Add fletch cmake directory for FindCUDNN.cmake needed by DarknetConfig.cmake
list( INSERT CMAKE_MODULE_PATH 0 "${VIAME_PACKAGES_DIR}/fletch/CMake" )

find_package( Darknet REQUIRED )

set( headers
  darknet_custom_resize.h
  darknet_detector.h
  darknet_trainer.h
)

set( sources
  darknet_custom_resize.cxx
  darknet_detector.cxx
  darknet_trainer.cxx
)

set( darknet_linked_libs
  Darknet::dark
  kwiver::kwiversys
  viame_opencv_bridge
  ${OpenCV_LIBS} )

add_definitions( -DOPENCV )

if( VIAME_ENABLE_CUDA )
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/include )
  add_definitions( -DDARKNET_USE_GPU )
endif()

if( VIAME_ENABLE_CUDNN )
  include_directories( ${CUDNN_INCLUDE_DIR} )
  add_definitions( -DDARKNET_USE_CUDNN )
endif()

if( WIN32 )
  add_definitions( -D_TIMESPEC_DEFINED )

  if( VIAME_ENABLE_CUDA )
    set( darknet_linked_libs
         ${darknet_linked_libs}
         ${CUDA_CUDA_LIBRARY}
         ${CUDA_LIBRARIES}
         ${CUDA_CUBLAS_LIBRARIES}
         ${CUDA_curand_LIBRARY} )
  endif()

  if( VIAME_ENABLE_CUDNN )
    set( darknet_linked_libs
         ${darknet_linked_libs}
         ${CUDNN_LIBRARIES} )
  endif()
endif()

kwiver_install_headers(
  SUBDIR     viame
  ${headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_darknet_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_darknet
  ${headers}
  ${sources}
  )

target_include_directories( viame_darknet PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

# darknet's and libsvm's headers, which come from the dependency tree this
# build takes fletch from; phase 1 vendors them. SYSTEM, so that the prefix
# reaches the compiler as -isystem and lands after every -I: VIAME installs
# its own headers into the same prefix, and an -I would let the copy left
# there by the last install shadow the tree being built.
target_include_directories( viame_darknet SYSTEM PUBLIC
  $<BUILD_INTERFACE:${VIAME_DEPENDENCY_INCLUDE_DIRS}>
  )

target_link_libraries( viame_darknet
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
  PRIVATE              ${darknet_linked_libs}
  )

set_target_properties( viame_darknet PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_darknet
  register_algorithms.cxx
  )

target_link_libraries( viame_darknet_plugin
  PUBLIC               kwiver::vital_vpm
  )
