# Build/install plugin for darknet CNN support
#

set( headers
  darknet_custom_resize.h
  darknet_detector.h
  darknet_trainer.h )

set( sources
  darknet_custom_resize.cxx
  darknet_detector.cxx
  darknet_trainer.cxx )

set( darknet_linked_libs
  darknet
  kwiversys
  kwiver_algo_ocv
  ${OpenCV_LIBS} )

# needed for certain headers
add_definitions( -DOPENCV )               # needed by darknet detector

# if darknet was built with the GPU support
if( Darknet_USING_GPU )
  find_package( CUDA QUIET REQUIRED )
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/include )
  add_definitions( -DDARKNET_USE_GPU )    # needed by darknet detector
endif()

if( Darknet_USING_CUDNN )
  include_directories( ${CUDNN_ROOT_DIR}/include )
  add_definitions( -DDARKNET_USE_CUDNN )  # needed by darknet detector
endif()


# Visual Studio links to a static lib of darknet due to the pthread dependency
# So we need to add all libraries to this arrow, if darknet was a dll, we would
# not have this issue, but we need pthreads and it likes static for some reason
if( WIN32 )
  add_definitions( -D_TIMESPEC_DEFINED )
  set( darknet_linked_libs ${darknet_linked_libs} pthreadVC2 )

  if( Darknet_USING_GPU )
    set( darknet_linked_libs
         ${darknet_linked_libs}
         ${CUDA_CUDA_LIBRARY}
         ${CUDA_LIBRARIES}
         ${CUDA_CUBLAS_LIBRARIES}
         ${CUDA_curand_LIBRARY} )
  endif()
  if( Darknet_USING_CUDNN )
    if( CUDNN_LIBRARY AND EXISTS ${CUDNN_LIBRARY} )
      set( CUDNN_LIBRARIES "${CUDNN_LIBRARY}" )
    else()
      message( STATUS "Darkent Arrow using CUDNN, looking for it here : ${CUDNN_ROOT_DIR}" )
      find_library( CUDNN_LIBRARIES
      NAMES cudnn libcudnn
      HINTS ${CUDNN_ROOT_DIR}
      PATH_SUFFIXES lib lib64 lib/x64 lib/x86 targets/aarch64-linux
      )
    endif()

    if( NOT CUDNN_LIBRARIES )
      message( FATAL_ERROR "Unable to find cudnn libraries" )
    endif()
    include_directories( ${CUDNN_ROOT_DIR}/include )
    set( darknet_linked_libs
         ${darknet_linked_libs}
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
  PRIVATE              ${darknet_linked_libs}
  )

algorithms_create_plugin( kwiver_algo_darknet
  register_algorithms.cxx
  )

if( KWIVER_ENABLE_TESTS )
#  add_subdirectory(tests)
endif()

if( KWIVER_ENABLE_PYTHON )
  kwiver_create_python_init( kwiver/arrows/darknet )

  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/generate_headers.py
    kwiver/arrows/darknet
    generate_headers )
endif()
