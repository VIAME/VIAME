# Included only for an explicit opt-in. Keep the CUDA runtime out of libviame
# and out of the ordinary Python image_kernels extension.
if( CMAKE_VERSION VERSION_LESS 3.18 )
  message( FATAL_ERROR "VIAME_ENABLE_CUDA_KERNELS requires CMake 3.18 or newer" )
endif()
enable_language( CUDA )
find_package( CUDAToolkit REQUIRED )
find_package( Threads REQUIRED )
include( GenerateExportHeader )
add_library( viame_image_kernels_cuda SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/filter.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/denoise.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/temporal.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/resample.cu )
add_library( viame::image_kernels_cuda ALIAS viame_image_kernels_cuda )
generate_export_header( viame_image_kernels_cuda )
target_include_directories( viame_image_kernels_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include/viame> )
target_link_libraries( viame_image_kernels_cuda PRIVATE CUDA::cudart Threads::Threads )
target_compile_features( viame_image_kernels_cuda PUBLIC cxx_std_17 )
target_compile_options( viame_image_kernels_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false> )
set_target_properties( viame_image_kernels_cuda PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Shared
  EXPORT_NAME image_kernels_cuda
  VERSION 1.0 SOVERSION 1
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  INSTALL_RPATH "$ORIGIN" )
install( TARGETS viame_image_kernels_cuda EXPORT viame-cuda-targets
  LIBRARY DESTINATION lib RUNTIME DESTINATION bin ARCHIVE DESTINATION lib )
install( EXPORT viame-cuda-targets NAMESPACE viame:: DESTINATION lib/cmake/viame )
install( FILES "${CMAKE_CURRENT_SOURCE_DIR}/cuda.h" "${CMAKE_CURRENT_BINARY_DIR}/viame_image_kernels_cuda_export.h"
  DESTINATION include/viame/image_kernels )
if( VIAME_ENABLE_PYTHON )
  viame_add_python_library( _cuda image_kernels SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_python.cxx
    PRIVATE viame_image_kernels_cuda )
  # VIAME enables --no-undefined on ELF modules; Windows also needs the
  # Python import library. This extension intentionally does not get it
  # transitively through libviame. macOS resolves Python in the interpreter.
  if( NOT APPLE )
    target_link_libraries( python-image_kernels-_cuda PRIVATE ${PYTHON_LIBRARIES} )
  endif()
endif()
