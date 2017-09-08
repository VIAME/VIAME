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

set( darknet_linked_libs
  ${darknet_LIBRARIES}
  kwiversys
  ${OpenCV_LIBS}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  kwiver_algo_ocv )

# if darknet was built with the GPU support
if (DARKNET_USE_GPU)
  find_package(CUDA QUIET REQUIRED)
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/include )
  add_definitions( -DDARKNET_USE_GPU )  # needed by darknet includes
endif()

# Visual Studio fix until project has better exports
if (WIN32)
  include_directories( SYSTEM ${darknet_INCLUDE_DIR}/../3rdparty/include )
  add_definitions( -D_TIMESPEC_DEFINED )
  set( darknet_linked_libs ${darknet_linked_libs} pthreadVC2 )
  if (KWIVER_ENABLE_CUDA)
    set( darknet_linked_libs
         ${darknet_linked_libs}
         ${CUDA_CUDA_LIBRARY}
         ${CUDA_LIBRARIES}
         ${CUDA_CUBLAS_LIBRARIES}
         ${CUDA_curand_LIBRARY} )
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
  set(kwiver_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
  set(kwiver_python_output_path "${KWIVER_BINARY_DIR}/lib/${kwiver_python_subdir}")

  kwiver_create_python_plugin_init( kwiver/arrows/darknet )

  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/generate_headers.py
    kwiver/arrows/darknet
    generate_headers )
endif()
