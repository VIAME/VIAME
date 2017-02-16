# Build/install plugin for darknet CNN support
#

set( headers
  darknet_detector.h
)

set( sources
  darknet_detector.cxx
  )

# if darknet was built with the GPU support
if (DARKNET_USE_GPU)
  find_package(CUDA QUIET REQUIRED)
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/include )
  add_definitions( -DDARKNET_USE_GPU )  # needed by darknet includes
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
  PRIVATE              ${darknet_LIBRARIES}
                       kwiversys
                       ${OpenCV_LIBS}
                       kwiver_algo_ocv
  )

algorithms_create_plugin( kwiver_algo_darknet
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
#  add_subdirectory(tests)
endif()
