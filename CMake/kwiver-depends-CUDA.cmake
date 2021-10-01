# Optionally find and configure CUDA dependency

option( KWIVER_ENABLE_CUDA
  "Enable CUDA dependent code and plugins"
  OFF
  )

if( KWIVER_ENABLE_CUDA )
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  endif()

  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    cmake_policy(SET CMP0104 NEW)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 35 50 52 60 61)
      if(CUDA_VERSION VERSION_GREATER_EQUAL "9.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 70)
      endif()
      if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 75)
      endif()
      if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 80)
      endif()
    endif()
  endif()
  message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif( KWIVER_ENABLE_CUDA )
