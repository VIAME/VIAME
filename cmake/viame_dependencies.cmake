###
# What VIAME's build needs to find
#
# Lifted out of the top-level `CMakeLists.txt` by P1-T04, unchanged. What is
# striking about it is how little is left: `lite-build-system.md` section 1
# tabulates this file by phase, and after P7 the row reads Python, Threads,
# optional OpenMP and optional CUDA. There is no `find_package( OpenCV )`,
# no FFMPEG, no Eigen, no ZLIB and no fletch -- each went with the phase
# that removed the dependency, and `library/tpl/` carries the small ones.
#
# What remains is CUDA and cuDNN (for darknet, and for telling pytorch what
# to install), python and its version checks, the pytorch version matrix,
# and two optional vendor libraries. The compiler check comes first because
# a partial C++17 is not a dependency anyone can install their way out of.
##

# Check for compilers with only partial C++17 support
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0" )
  link_libraries( stdc++fs )
endif()

# Check CUDA related paths
if( VIAME_ENABLE_CUDA )
  find_package( CUDA QUIET REQUIRED )

  # Resolve symlinks in CUDA_NVCC_EXECUTABLE to avoid CMake 3.31+ detection issues
  # When nvcc is invoked via a symlink (e.g., /usr/bin/nvcc -> /usr/local/cuda/bin/nvcc),
  # it reports incorrect paths that break CMake's CUDA toolkit detection
  if( CUDA_NVCC_EXECUTABLE AND IS_SYMLINK "${CUDA_NVCC_EXECUTABLE}" )
    get_filename_component( CUDA_NVCC_EXECUTABLE "${CUDA_NVCC_EXECUTABLE}" REALPATH )
    set( CUDA_NVCC_EXECUTABLE "${CUDA_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA nvcc executable" FORCE )
  endif()

  # `library/tpl/darknet` compiles CUDA in this build now, which needs the
  # first-class language rather than the old `find_package( CUDA )` module.
  # It has to be told the **resolved** nvcc for the same reason as above:
  # handed `/usr/bin/nvcc`, CMake 4 cannot work back to the toolkit root and
  # fails `enable_language( CUDA )` with "Couldn't find CUDA library root".
  if( NOT CMAKE_CUDA_COMPILER OR IS_SYMLINK "${CMAKE_CUDA_COMPILER}" )
    set( CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}" CACHE FILEPATH
         "CUDA compiler" FORCE )
  endif()

  if( NOT DEFINED CMAKE_CUDA_ARCHITECTURES )
    # VIAME's own `CUDA_ARCHITECTURES` is set further down as dotted versions
    # -- "6.0 6.1 7.0 ..." -- which is the form the superbuild passed to
    # darknet after stripping the dots. `CMAKE_CUDA_ARCHITECTURES` wants the
    # same list without them, so it is derived there rather than written out
    # twice; this is only the fallback for a build that sets neither.
    #
    # **Version-guarded for the same reason the dotted list below is.** CUDA
    # 13 removed Maxwell, Pascal and Volta. This value is what
    # `enable_language( CUDA )` compiles its own test with, and that happens
    # before the dotted list is computed, so it cannot borrow the guard from
    # there. Left at the old list a CUDA 13 configure fails as
    #
    #     nvcc fatal : Unsupported gpu architecture 'compute_60'
    #
    # which reads as a broken toolkit rather than as an architecture VIAME
    # asked for and CUDA no longer has. `CUDA_VERSION` is available here:
    # `find_package( CUDA )` runs twenty lines above.
    if( CUDA_VERSION VERSION_LESS "13.0" )
      set( CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90" )
    else()
      set( CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" )
    endif()
  endif()

  if( CUDA_VERSION_MAJOR GREATER_EQUAL 10 AND NOT CUDA_cublas_device_LIBRARY )
    set( CUDA_cublas_device_LIBRARY CACHE INTERNAL "${CUDA_cublas_LIBRARY}" )
  endif()

  if( NOT CUDA_VERSION_PATCH )
    if( CUDA_NVCC_EXECUTABLE AND
        CUDA_NVCC_EXECUTABLE STREQUAL CMAKE_CUDA_COMPILER AND
        CMAKE_CUDA_COMPILER_VERSION MATCHES [=[([0-9]+)\.([0-9]+)\.([0-9]+)]=] )
      set( CUDA_VERSION_PATCH "${CMAKE_MATCH_3}" )
    elseif( CUDA_NVCC_EXECUTABLE )
      execute_process( COMMAND ${CUDA_NVCC_EXECUTABLE} "--version" OUTPUT_VARIABLE NOUT )
      if( NOUT MATCHES [=[ V([0-9]+)\.([0-9]+)\.([0-9]+)]=] )
        set( CUDA_VERSION_PATCH "${CMAKE_MATCH_3}" )
      endif()
    endif()
  endif()

  if( CUDA_VERSION_MAJOR VERSION_LESS "10" )
    message( FATAL_ERROR "VIAME does not support CUDA versions under v10.0" )
  endif()

  if( CUDA_VERSION VERSION_EQUAL "10.1" AND CUDA_VERSION_PATCH EQUAL "168" )
    message( FATAL_ERROR "CUDA 10.1.168 has bugs, upgrade to 10.1.264 or above" )
  endif()

  # `VIAME_FORCE_CUDA_CSTD98` stood here, and was read in one place: what
  # C++ standard fletch compiled its CUDA with.

  # CUDA 13.0 removed support for Maxwell, Pascal, and Volta archs
  if( CUDA_VERSION VERSION_LESS "13.0" )
    set( DEF_CUDA_ARCHS "6.0 6.1 7.0 7.5" )
  else()
    set( DEF_CUDA_ARCHS "7.5" )
  endif()

  if( CUDA_VERSION VERSION_LESS "12.0" )
    set( DEF_CUDA_ARCHS "3.5 ${DEF_CUDA_ARCHS}" )
  endif()
  if( CUDA_VERSION VERSION_GREATER "10.5" )
    set( DEF_CUDA_ARCHS "${DEF_CUDA_ARCHS} 8.0" )
    if( CUDA_VERSION VERSION_LESS "11.1" )
      set( DEF_CUDA_ARCHS "${DEF_CUDA_ARCHS} 8.0+PTX" )
    endif()
  endif()
  if( CUDA_VERSION VERSION_GREATER "11.0" )
    set( DEF_CUDA_ARCHS "${DEF_CUDA_ARCHS} 8.6" )
    if( CUDA_VERSION VERSION_LESS "12.1" )
      set( DEF_CUDA_ARCHS "${DEF_CUDA_ARCHS} 8.6+PTX" )
    endif()
  endif()
  if( CUDA_VERSION VERSION_GREATER "11.7" )
    set( DEF_CUDA_ARCHS "${DEF_CUDA_ARCHS} 8.9 9.0" )
  endif()
  if( CUDA_VERSION VERSION_GREATER "12.7" )
    set( DEF_CUDA_ARCHS "${DEF_CUDA_ARCHS} 10.0 12.0" )
  endif()

  set( CUDA_ARCHITECTURES "${DEF_CUDA_ARCHS}" CACHE STRING "CUDA Architectures" )
  mark_as_advanced( CUDA_ARCHITECTURES )

  # CUDA 12.x does not officially support VS versions after 2022, pass flag
  # to nvcc to allow compilation with newer host compilers (e.g. VS 2025)
  if( WIN32 )
    set( CMAKE_CUDA_FLAGS "--allow-unsupported-compiler ${CMAKE_CUDA_FLAGS}"
      CACHE STRING "CUDA compilation flags" FORCE )
  endif()

  if( WIN32 )
    set( VIAME_EXECUTABLES_PATH
      ${VIAME_BUILD_INSTALL_PREFIX}/bin;${CUDA_TOOLKIT_ROOT_DIR}/bin;$ENV{PATH}
      CACHE INTERNAL "All compiled and system-related runnable executables" )
  else()
    set( VIAME_EXECUTABLES_PATH
      ${VIAME_BUILD_INSTALL_PREFIX}/bin:${CUDA_TOOLKIT_ROOT_DIR}/bin:$ENV{PATH}
      CACHE INTERNAL "All compiled and system-related runnable executables" )
  endif()

  if( CMAKE_COMPILER_IS_GNUCC AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0 )
    message( WARNING "GCC 14.0+ support is experimental and may not work." )
  endif()

  if( CMAKE_COMPILER_IS_GNUCC AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 12.0 AND
      CUDA_VERSION VERSION_LESS "11.6" )
    message( FATAL_ERROR "If using GCC 12.0 or greater, enable CUDA 11.7 or "
      "greater, alongside pytorch 1.12 or greater unless you know enough to "
      "hack around this requirement and disable this error." )
  endif()
else()
  set( CUDA_ARCHITECTURES "" CACHE INTERNAL "CUDA Architectures" )

  if( WIN32 )
    set( VIAME_EXECUTABLES_PATH
      ${VIAME_BUILD_INSTALL_PREFIX}/bin;$ENV{PATH}
      CACHE INTERNAL "All compiled and system-related runnable executables" )
  else()
    set( VIAME_EXECUTABLES_PATH
      ${VIAME_BUILD_INSTALL_PREFIX}/bin:$ENV{PATH}
      CACHE INTERNAL "All compiled and system-related runnable executables" )
  endif()
endif()

if( VIAME_ENABLE_CUDNN AND NOT VIAME_ENABLE_CUDA )
  message( FATAL_ERROR "Cannot enable CUDNN without CUDA, disable VIAME_ENABLE_CUDNN" )
endif()

if( VIAME_ENABLE_CUDNN )
  set( CUDNN_ROOT_DIR "" CACHE PATH "CUDNN root folder, leave as blank to auto-detect." )

  if( WIN32 )
    set( DEFAULT_SYS_CUDNN_DIR "C:/Program Files/NVIDIA/CUDNN/v9.10" )
  else()
    set( DEFAULT_SYS_CUDNN_DIR /usr )
  endif()

  # The active CUDA toolkit (CUDA_TOOLKIT_ROOT_DIR) may not be the one shipping
  # cudnn (e.g. /usr/local/cuda -> cuda-12.9 with no cudnn, while cudnn lives in
  # cuda-12.6). Glob the sibling versioned CUDA install dirs so cudnn is still
  # auto-detected in that case.
  set( CUDNN_VERSIONED_HINTS )
  if( NOT WIN32 )
    get_filename_component( CUDA_PARENT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY )
    file( GLOB CUDNN_VERSIONED_HINTS
      "${CUDA_PARENT_DIR}/cuda-*/targets/x86_64-linux"
      "${CUDA_PARENT_DIR}/cuda-*/targets/aarch64-linux" )
  endif()

  find_library( CUDNN_LIBRARY REQUIRED
    NAMES cudnn cudnn64 libcudnn.so libcudnn.so.9 libcudnn.so.8
    HINTS ${CUDNN_ROOT_DIR}
          ${CUDNN_ROOT_DIR}/..
          ${CUDNN_ROOT_DIR}/lib/x64
          ${CUDA_TOOLKIT_ROOT_DIR}
          ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux
          ${CUDNN_VERSIONED_HINTS}
          ${DEFAULT_SYS_CUDNN_DIR}
          ${DEFAULT_SYS_CUDNN_DIR}/lib/x86_64-linux-gnu
    PATH_SUFFIXES lib lib64 )

  get_filename_component( CUDNN_LIBRARY "${CUDNN_LIBRARY}" REALPATH )

  if( NOT CUDNN_LIBRARY )
    message( FATAL_ERROR "Unable to locate CUDNN library" )
  endif()

  # Different subprojects use each variable
  set( CUDNN_LIBRARIES "${CUDNN_LIBRARY}" CACHE INTERNAL "" FORCE )

  # Check version of CUDNN
  get_filename_component( CUDNN_ROOT_DIR_TMP "${CUDNN_LIBRARY}" DIRECTORY )
  get_filename_component( CUDNN_ROOT_DIR_TMP "${CUDNN_ROOT_DIR_TMP}" DIRECTORY )

  set( CUDNN_ROOT_DIR "${CUDNN_ROOT_DIR_TMP}" CACHE INTERNAL "CUDNN root folder" FORCE )
  set( CUDNN_INCLUDE_FILE "${CUDNN_ROOT_DIR}/include/cudnn.h" CACHE INTERNAL "" FORCE )

  if( CUDNN_ROOT_DIR STREQUAL "/" )
    set( CUDNN_INCLUDE_FILE "/include/cudnn.h" CACHE INTERNAL "" FORCE )
  endif()

  if( NOT EXISTS ${CUDNN_INCLUDE_FILE} )
    get_filename_component( CUDNN_ROOT_DIR_TMP "${CUDNN_ROOT_DIR_TMP}" DIRECTORY )

    set( CUDNN_ROOT_DIR "${CUDNN_ROOT_DIR_TMP}" CACHE INTERNAL "CUDNN root folder" FORCE )
    set( CUDNN_INCLUDE_FILE "${CUDNN_ROOT_DIR}/include/cudnn.h" CACHE INTERNAL "" FORCE )
  endif()

  if( NOT EXISTS ${CUDNN_INCLUDE_FILE} )
    set( CUDNN_INCLUDE_FILE "${CUDNN_ROOT_DIR}/include/x86_64-linux-gnu/cudnn.h" CACHE INTERNAL "" FORCE )
  endif()

  if( NOT EXISTS ${CUDNN_INCLUDE_FILE} )
    message( FATAL_ERROR "Unable to locate cudnn.h include header" )
  endif()

  get_filename_component( CUDNN_INCLUDE_DIR "${CUDNN_INCLUDE_FILE}" DIRECTORY )

  if( EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h" )
    set( CUDNN_VERSION_FILE "${CUDNN_INCLUDE_DIR}/cudnn_version.h" )
  else()
    set( CUDNN_VERSION_FILE "${CUDNN_INCLUDE_FILE}" )
  endif()

  if( EXISTS "${CUDNN_VERSION_FILE}" )
    file( READ "${CUDNN_VERSION_FILE}" TMP_CUDNN_VERSION_FILE_CONTENTS )

    string( REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
            CUDNN_VERSION_MAJOR "${TMP_CUDNN_VERSION_FILE_CONTENTS}" )
    string( REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
            CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}" )
  endif()
endif()

if( VIAME_ENABLE_PYTHON )
  if( NOT WIN32 )
    option( VIAME_PYTHON_SYMLINK "Symlink python files instead of copying." OFF )
    mark_as_advanced( VIAME_PYTHON_SYMLINK )
  endif()
  set( KWIVER_SYMLINK_PYTHON ${VIAME_PYTHON_SYMLINK} ) # Required for KWIVER scripts

  # Should we build a version of python within VIAME itself
  option( VIAME_BUILD_PYTHON_FROM_SOURCE "Build the actual CPython interpreter" OFF )
  mark_as_advanced( VIAME_BUILD_PYTHON_FROM_SOURCE )

  # Or download a pinned, relocatable one into the install; see
  # cmake/viame_python_standalone.cmake
  option( VIAME_PYTHON_STANDALONE
    "Download a pinned python-build-standalone CPython into the install and build against it" OFF )

  if( VIAME_PYTHON_STANDALONE AND VIAME_BUILD_PYTHON_FROM_SOURCE )
    message( FATAL_ERROR "VIAME_PYTHON_STANDALONE and VIAME_BUILD_PYTHON_FROM_SOURCE "
      "both provide the install's python; turn one of them off." )
  endif()

  if( VIAME_BUILD_PYTHON_FROM_SOURCE )
    set( VIAME_PYTHON_VERSION 3.12.12
         CACHE STRING "Select the version of Python to build." )
    set_property( CACHE VIAME_PYTHON_VERSION
                  PROPERTY STRINGS "3.10.4" "3.12.12" "3.14.2" "3.14.2t" )
    mark_as_advanced( VIAME_PYTHON_VERSION )

    # Jobs for CPython's own make; its configure is autotools and knows
    # nothing of the generator's parallelism
    include( ProcessorCount )
    ProcessorCount( VIAME_PYTHON_BUILD_JOBS )
    if( VIAME_PYTHON_BUILD_JOBS EQUAL 0 )
      set( VIAME_PYTHON_BUILD_JOBS 1 )
    endif()

    include( setup_internal_python )
  elseif( VIAME_PYTHON_STANDALONE )
    include( viame_python_standalone )
    find_package( Python ${VIAME_PYTHON_STANDALONE_VERSION} EXACT
      COMPONENTS Interpreter Development REQUIRED )
  else()
    find_package( Python COMPONENTS Interpreter Development REQUIRED )
  endif()

  if( VIAME_ENABLE_PYTORCH-SLEAP )
    if( Python_VERSION VERSION_LESS "3.11" OR NOT Python_VERSION VERSION_LESS "3.14" )
      message( FATAL_ERROR "SLEAP-NN v0.3.3 requires Python 3.11 through 3.13" )
    endif()
    if( NOT VIAME_ENABLE_PYTORCH-VISION OR NOT VIAME_ENABLE_OPENCV )
      message( FATAL_ERROR "SLEAP-NN requires VIAME_ENABLE_PYTORCH-VISION and VIAME_ENABLE_OPENCV" )
    endif()
  endif()

  # Backwards compatibility for sub-projects which use "PYTHON_" cmake
  # variables and the old find_package( PythonInterp ) commands instead
  # of the newer find Python. Copies all Python_* to PYTHON_* vars.
  CopyVarsToAllCaps( "Python" )

  if( VIAME_ENABLE_PYTHON-NETHARN AND NOT VIAME_ENABLE_PYTHON-MMDET )
    message( FATAL_ERROR "Netharn currently also requires mmdet enabled" )
  endif()

  set( VIAME_PYTHON_STRING "python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}"
    CACHE INTERNAL "Version ID used in python install paths" )

  set( PYBASE ${VIAME_BUILD_INSTALL_PREFIX}/lib/${VIAME_PYTHON_STRING} )
  set( VIAME_PYTHON_INSTALL ${PYBASE} CACHE INTERNAL "VIAME Python install path" )
  set( VIAME_PYTHON_PACKAGES ${PYBASE}/site-packages CACHE INTERNAL "Internal site-packages" )
  set( VIAME_PYTHON_USERBASE "${VIAME_INSTALL_PREFIX}" )

  # Add PYTHON_PATH variables internal to the VIAME build tree, e.g. things installed
  # just for VIAME not in an external Python environment or install
  set( PYTHON_PATH_DESC "Pythonpath for all files installed as a part of VIAME" )
  if( WIN32 )
    string( REPLACE "/" "\\" VIAME_PYTHON_USERBASE "${VIAME_PYTHON_USERBASE}" )
    set( VIAME_PYTHON_PATH
      ${PYBASE};${PYBASE}/site-packages;${PYBASE}/dist-packages
      CACHE INTERNAL ${PYTHON_PATH_DESC} )
    if( NOT VIAME_BUILD_PYTHON_FROM_SOURCE )
      # If using system python on windows, add all possible path locations for
      # python packages in the system install just so they can be used in addition
      # to the ones within the VIAME install tree
      if( EXISTS "${PYTHON_STDLIB}" AND EXISTS "${PYTHON_SITELIB}" )
        set( VIAME_PYTHON_PATH
          ${VIAME_PYTHON_PATH};${PYTHON_STDLIB};${PYTHON_SITELIB}
          CACHE INTERNAL ${PYTHON_PATH_DESC} )
       endif()
      if( EXISTS "${PYTHON_RUNTIME_LIBRARY_DIRS}/DLLs" )
        set( VIAME_PYTHON_PATH
          ${VIAME_PYTHON_PATH};${PYTHON_RUNTIME_LIBRARY_DIRS}/DLLs
          CACHE INTERNAL ${PYTHON_PATH_DESC} )
      endif()
    endif()
  else()
    set( VIAME_PYTHON_PATH
      ${PYBASE}:${PYBASE}/site-packages:${PYBASE}/dist-packages
      CACHE INTERNAL ${PYTHON_PATH_DESC} )
  endif()

  # Configure PYTHON_DEP_ENV_VARS, this is used to set the environment used for
  # either building or installing python dependencies
  set( PYTHON_DEP_ENV_VARS )

  if( WIN32 )
    set( ADJ_INCLUDE_PATH "${VIAME_INSTALL_PREFIX}/include;$ENV{INCLUDE}" )
    set( ADJ_LIBRARY_PATH "${VIAME_INSTALL_PREFIX}/lib;${VIAME_INSTALL_PREFIX}/bin;$ENV{LIB}" )

    if( VIAME_BUILD_PYTHON_FROM_SOURCE )
      set( ENV{PYTHONPATH} "${VIAME_PYTHON_PATH};$ENV{PYTHONPATH}" )
      list( APPEND PYTHON_DEP_ENV_VARS "PYTHONHOME=${VIAME_PYTHON_USERBASE}" )
    else()
      if( EXISTS "${PYTHON_INCLUDE_DIRS}" )
        set( ADJ_INCLUDE_PATH "${ADJ_INCLUDE_PATH};${PYTHON_INCLUDE_DIRS}" )
      endif()
      if( EXISTS "${PYTHON_LIBRARY_DIRS}" )
        set( ADJ_LIBRARY_PATH "${ADJ_LIBRARY_PATH};${PYTHON_LIBRARY_DIRS}" )
      endif()
    endif()

    # Use <PS> as path separator instead of ---- to avoid conflict with LIST_SEPARATOR
    # <PS> will be converted back to ; (Windows) or : (Unix) by build scripts
    string( REPLACE ";" "<PS>" VIAME_PYTHON_PATH "${VIAME_PYTHON_PATH}" )
    string( REPLACE ";" "<PS>" VIAME_EXECUTABLES_PATH "${VIAME_EXECUTABLES_PATH}" )
    string( REPLACE ";" "<PS>" ADJ_INCLUDE_PATH "${ADJ_INCLUDE_PATH}" )
    string( REPLACE ";" "<PS>" ADJ_LIBRARY_PATH "${ADJ_LIBRARY_PATH}" )

    list( APPEND PYTHON_DEP_ENV_VARS "INCLUDE=${ADJ_INCLUDE_PATH}" )
    list( APPEND PYTHON_DEP_ENV_VARS "LIB=${ADJ_LIBRARY_PATH}" )
    list( APPEND PYTHON_DEP_ENV_VARS "PYTHONIOENCODING=UTF-8" )
  else()
    set( ADJ_LD_LIB_PATH "${VIAME_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}" )

    list( APPEND PYTHON_DEP_ENV_VARS "PATH=${VIAME_EXECUTABLES_PATH}" )
    list( APPEND PYTHON_DEP_ENV_VARS "CPPFLAGS=-I${VIAME_INSTALL_PREFIX}/include" )
    list( APPEND PYTHON_DEP_ENV_VARS "LDFLAGS=-L${VIAME_INSTALL_PREFIX}/lib" )
    list( APPEND PYTHON_DEP_ENV_VARS "CC=${CMAKE_C_COMPILER}" )
    list( APPEND PYTHON_DEP_ENV_VARS "CXX=${CMAKE_CXX_COMPILER}" )
    list( APPEND PYTHON_DEP_ENV_VARS "LD_LIBRARY_PATH=${ADJ_LD_LIB_PATH}" )
  endif()

  list( APPEND PYTHON_DEP_ENV_VARS "PYTHONPATH=${VIAME_PYTHON_PATH}" )
  list( APPEND PYTHON_DEP_ENV_VARS "PYTHONUSERBASE=${VIAME_PYTHON_USERBASE}" )
  list( APPEND PYTHON_DEP_ENV_VARS "PKG_CONFIG_PATH=${VIAME_INSTALL_PREFIX}/lib/pkgconfig" )

  # CMake 4.0+ removed compatibility with cmake_minimum_required( VERSION < 3.5 ).
  # PyTorch (and other python deps) bundle ancient third-party projects (NNPACK,
  # psimd, FP16, protobuf, etc.) that still declare such low minimums and would
  # otherwise abort configuration. This env var is honored by every (including
  # nested) cmake invocation and raises the policy floor to 3.5 for them.
  list( APPEND PYTHON_DEP_ENV_VARS "CMAKE_POLICY_VERSION_MINIMUM=3.5" )

  # PEP 668: distributions from Ubuntu 24.04 / Debian 12 onwards drop an
  # EXTERNALLY-MANAGED marker beside their system interpreter, which makes pip
  # refuse every install that is not into a venv, --target or --prefix. All of
  # the superbuild's python dependencies go in through "pip install --user"
  # with PYTHONUSERBASE pointed at VIAME's own install tree, so they are
  # refused outright and the build dies as soon as the first python dep runs.
  # Nothing lands in the distro's site-packages, so opt out of the marker.
  # Only set when the marker is actually present: the flag arrived in pip
  # 23.0.1, well after the older interpreters (22.04, Rocky) VIAME also builds
  # against, and an internally built python is never externally managed.
  if( NOT VIAME_BUILD_PYTHON_FROM_SOURCE )
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c
        "import os, sysconfig; print( os.path.exists( os.path.join( sysconfig.get_path( 'stdlib' ), 'EXTERNALLY-MANAGED' ) ) )"
      OUTPUT_VARIABLE VIAME_PYTHON_EXTERNALLY_MANAGED
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET )

    if( VIAME_PYTHON_EXTERNALLY_MANAGED STREQUAL "True" )
      message( STATUS "System python is PEP 668 externally managed, enabling "
        "PIP_BREAK_SYSTEM_PACKAGES for VIAME python dependency installs" )
      list( APPEND PYTHON_DEP_ENV_VARS "PIP_BREAK_SYSTEM_PACKAGES=1" )
    endif()
  endif()

  if( VIAME_BUILD_MAX_THREADS )
    list( APPEND PYTHON_DEP_ENV_VARS "MAX_JOBS=${VIAME_BUILD_MAX_THREADS}" )
  endif()
endif()

if( WIN32 AND VIAME_ENABLE_PYTHON AND CMAKE_CONFIGURATION_TYPES EQUAL "Debug" )
  message( FATAL_ERROR "Cannot build in Debug on Windows with Python enabled, \
    build in RelWithDebInfo or Release until fixed." )
endif()



if( VIAME_ENABLE_PYTORCH )
  # PyTorch and torchvision come as wheels from the index the accelerator lock
  # names; `VIAME_BUILD_PYTORCH_FROM_SOURCE` and its checks went in P1-T02.
  if( VIAME_PYTORCH_VERSION VERSION_EQUAL "${PYTORCH_INTERNAL_VERSION}" AND
      Python_VERSION VERSION_LESS "${PYTORCH_MIN_PYTHON_WHL}" )
    message( FATAL_ERROR "PyTorch ${VIAME_PYTORCH_VERSION} from pip requires at "
      "least python ${PYTORCH_MIN_PYTHON_WHL}." )
  endif()
  if( VIAME_ENABLE_CUDA )
    if( VIAME_PYTORCH_VERSION VERSION_EQUAL "2.12.0" AND
        NOT ( CUDA_VERSION VERSION_EQUAL "12.6" OR
              CUDA_VERSION VERSION_EQUAL "13.0" OR
              CUDA_VERSION VERSION_EQUAL "13.2" ) )
      message( FATAL_ERROR "CUDA 12.6, 13.0 or 13.2 is required for VIAME_ENABLE_PYTORCH "
        "with PyTorch 2.12.0. Either modify VIAME_PYTORCH_VERSION or the CUDA version." )
    elseif( VIAME_PYTORCH_VERSION VERSION_EQUAL "1.13.1" AND
        NOT ( CUDA_VERSION VERSION_EQUAL "11.6" OR
              CUDA_VERSION VERSION_EQUAL "11.7" ) )
      message( FATAL_ERROR "CUDA 11.7, or 11.6 is required for VIAME_ENABLE_PYTORCH "
        "with PyTorch 1.13.1. Either modify VIAME_PYTORCH_VERSION or the CUDA version." )
    endif()
  endif()
  if( Python_VERSION VERSION_LESS "3.6.2" )
    message( FATAL_ERROR "Only python distributions >= 3.6.2 are supported with "
      "pytorch enabled. If you think you have Python3.6+ installed, make sure you "
      "also have the python header package installed, e.g. python3-dev or "
      "python3-devel." )
  endif()
endif()

# `VIAME_OPENCV_VERSION` stood here, choosing which OpenCV fletch built, with
# a check that GCC 12 was not pointed at one older than 4.6. Nothing compiles
# against OpenCV any more and cv2 is a wheel, so the version is the wheel's.

if( VIAME_ENABLE_SEAGIS )
  option( VIAME_BUILD_SEAGIS_TEST_LIB "Use mock lib for testing only" OFF )

  set( SEAGIS_ROOT_DIR ""
       CACHE PATH "Path to SEAGIS StereoLibLX library" )

  mark_as_advanced( SEAGIS_ROOT_DIR )

  set( SEAGIS_INCLUDE_DIR "${SEAGIS_ROOT_DIR}"
       CACHE INTERNAL "Path to SEAGIS include directory" )

  if( VIAME_BUILD_SEAGIS_TEST_LIB )
    set( SEAGIS_LIBRARY "" CACHE INTERNAL "Not used in mock mode" )
  else()
    set( SEAGIS_LIBRARY "${SEAGIS_ROOT_DIR}/StereoLibLX_patched.lib"
         CACHE INTERNAL "Path to SEAGIS library file" )
  endif()

  if( NOT EXISTS "${SEAGIS_INCLUDE_DIR}/LX_StereoInterface.h" )
    message( FATAL_ERROR "SEAGIS include directory does not contain LX_StereoInterface.h. "
      "Please set SEAGIS_ROOT_DIR to the correct location." )
  endif()
  if( NOT VIAME_BUILD_SEAGIS_TEST_LIB AND NOT EXISTS "${SEAGIS_LIBRARY}" )
    message( FATAL_ERROR "SEAGIS library not found at ${SEAGIS_LIBRARY}. "
      "Please set SEAGIS_LIBRARY to the correct path." )
  endif()
endif()
