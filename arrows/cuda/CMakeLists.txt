
project(kwiver_arrow_cuda LANGUAGES CXX CUDA)

set(CMAKE_FOLDER "Arrows/CUDA")

set(header_files
  integrate_depth_maps.h
  )

set(private_header_files
  cuda_error_check.h
  cuda_memory.h
  )

set(source_files
  cuda_error_check.cxx
  integrate_depth_maps.cxx
  integrate_depth_maps.cu
  )

kwiver_install_headers(
  SUBDIR     arrows/cuda
  ${header_files}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_cuda_export.h
  NOPATH   SUBDIR     arrows/cuda
  )

kwiver_add_library( kwiver_algo_cuda
  ${header_files}
  ${private_header_files}
  ${source_files}
  )

set_target_properties(kwiver_algo_cuda
  PROPERTIES   POSITION_INDEPENDENT_CODE ON
               CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries( kwiver_algo_cuda
  PUBLIC               vital_algo
  PRIVATE              kwiver_algo_core
)

algorithms_create_plugin( kwiver_algo_cuda
  register_algorithms.cxx
  )
