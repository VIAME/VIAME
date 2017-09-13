# Build/install plugin for darknet CNN support
#

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

set( Darknet_linked_libs
  ${Darknet_LIBRARIES}
  kwiversys
  ${OpenCV_LIBS}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  kwiver_algo_ocv )

# if darknet was built with the GPU support
if (Darknet_USING_GPU)
  find_package(CUDA QUIET REQUIRED)
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/include )
  add_definitions( -DDARKNET_USE_GPU )  # needed by darknet detector
endif()
if (Darknet_USING_CUDNN)
  find_library( CUDNN_LIBRARIES
    NAMES cudnn libcudnn
    HINTS ${CUDNN_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86 targets/aarch64-linux
  )
  if(NOT CUDNN_LIBRARIES)
    message(FATAL_ERROR "Unable to find cudnn libraries")
  endif()
  include_directories( ${CUDNN_ROOT_DIR}/include )
endif()

# Visual Studio fix until project has better exports
if (WIN32)
  include_directories( SYSTEM ${Darknet_INCLUDE_DIR}/../3rdparty/include )
  add_definitions( -D_TIMESPEC_DEFINED )
  set( Darknet_linked_libs ${Darknet_linked_libs} pthreadVC2 )
  if (Darknet_USING_GPU)
    set( Darknet_linked_libs
         ${Darknet_linked_libs}
         ${CUDA_CUDA_LIBRARY}
         ${CUDA_LIBRARIES}
         ${CUDA_CUBLAS_LIBRARIES}
         ${CUDA_curand_LIBRARY} )
  endif()
  if (Darknet_USING_CUDNN)
    set( Darknet_linked_libs
         ${Darknet_linked_libs}
         ${CUDNN_LIBRARIES} )
  endif()
endif()

kwiver_install_headers(
  SUBDIR     arrows/darknet
  ${headers}
  )
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_darknet_export.h
  NOPATH   SUBDIR     arrows/darknet
  )


kwiver_add_library( kwiver_algo_darknet
  ${headers}
  ${sources}
  )

target_link_libraries( kwiver_algo_darknet
  PUBLIC               vital_algo
  PRIVATE              ${Darknet_linked_libs}
  )

algorithms_create_plugin( kwiver_algo_darknet
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
#  add_subdirectory(tests)
endif()
