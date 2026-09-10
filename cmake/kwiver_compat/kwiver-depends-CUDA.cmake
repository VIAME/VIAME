# Optionally find and configure CUDA dependency

option( KWIVER_ENABLE_CUDA
  "Enable CUDA dependent code and plugins"
  OFF
  )

if(DEFINED fletch_BUILT_WITH_CUDNN)
  set(_default ${fletch_BUILT_WITH_CUDNN})
else()
  set(_default OFF)
endif()
option( KWIVER_ENABLE_CUDNN
  "Enable CUDNN support (requires KWIVER_ENABLE_CUDA)"
  ${_default}
  )
unset(_default)

if( KWIVER_ENABLE_CUDA )
  # Set CMAKE_CUDA_COMPILER from CUDA_NVCC_EXECUTABLE if available
  # This is needed for CMake 3.31+ which has different CUDA detection
  if( CUDA_NVCC_EXECUTABLE AND NOT CMAKE_CUDA_COMPILER )
    set( CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" )
  endif()
  if( CUDA_TOOLKIT_ROOT_DIR AND NOT CUDAToolkit_ROOT )
    set( CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "CUDA toolkit root" )
  endif()

  # Find CUDA toolkit first to help CMake locate CUDA
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND AND NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler")
  endif()

  enable_language(CUDA)

  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
  elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    cmake_policy(SET CMP0104 NEW)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
        set(CMAKE_CUDA_ARCHITECTURES 52 60 61)
      else()
        set(CMAKE_CUDA_ARCHITECTURES 35 50 52 60 61)
      endif()
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "9.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 70)
      endif()
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "10.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 75)
      endif()
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 80)
      endif()
    endif()
  endif()
  message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif( KWIVER_ENABLE_CUDA )
