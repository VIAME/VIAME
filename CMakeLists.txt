#
# Top level build script for kwiver-build
#
cmake_minimum_required(VERSION 2.8.3)
enable_testing()

project(KWIVER)

include(CMakeDependentOption)

###
# KWIVER version
set(KWIVER_VERSION_MAJOR 0)
set(KWIVER_VERSION_MINOR 1)
set(KWIVER_VERSION_PATCH 0)
set(KWIVER_VERSION "${KWIVER_VERSION_MAJOR}.${KWIVER_VERSION_MINOR}.${KWIVER_VERSION_PATCH}")

###
# project specific directories
set(KWIVER_PACKAGES_DIR "${CMAKE_SOURCE_DIR}/Packages")
set(KWIVER_PACKAGING_SOURCE_DIR "${CMAKE_SOURCE_DIR}/KWIVER_packaging")
set(KWIVER_PACKAGING_BINARY_DIR "${CMAKE_BINARY_DIR}/KWIVER_packaging")
set(KWIVER_CMAKE_DIR "${CMAKE_SOURCE_DIR}/CMake")

set(KWIVER_BUILD_PREFIX "${CMAKE_BINARY_DIR}/build")
set(KWIVER_BUILD_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")

set(CMAKE_MODULE_PATH ${KWIVER_CMAKE_DIR} ${CMAKE_MODULE_PATH})
set(CMAKE_FIND_ROOT_PATH ${KWIVER_BUILD_INSTALL_PREFIX} ${CMAKE_FIND_ROOT_PATH})
set(CMAKE_LIBRARY_PATH ${KWIVER_BUILD_INSTALL_PREFIX}/lib)
set(KWIVER_LINKER_FLAGS -L${CMAKE_LIBRARY_PATH})

# ================================================================
option(KWIVER_ENABLE_OPENCV "Enable OpenCV-dependent code" ON )
option(Boost_USE_STATIC_LIBS "Use a statically-linked Boost" OFF)

set(BOOST_ROOT "" CACHE PATH "Location of boost install")

find_package(Boost 1.48 REQUIRED
  COMPONENTS
    chrono
    date_time
    filesystem
    program_options
    system
    thread)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

set(BOOST_root ${Boost_LIBRARY_DIR}/..)

# ================================================================
# project global includes
include_directories(BEFORE "${KWIVER_BUILD_INSTALL_PREFIX}/include")


if(NOT WIN32)
  option(KWIVER_BUILD_SHARED "Build KWIVER packages shared or not" ON)
else()
  set(KWIVER_BUILD_SHARED OFF)
endif()

if( KWIVER_ENABLE_OPENCV )
  find_package( OpenCV 2.4.6 REQUIRED )
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
endif( KWIVER_ENABLE_OPENCV )
set( USE_OPENCV_VERSION KWIVER_ENABLE_OPENCV AND OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "2.4" )

if(KWIVER_BUILD_SHARED)
  set(LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
  set(LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
else()
  set(LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
  set(LIB_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

set(BUILD_SHARED_LIBS ${KWIVER_BUILD_SHARED})

# installing system setup script
configure_file(${KWIVER_CMAKE_DIR}/setup_KWIVER.sh.in
  ${KWIVER_BUILD_INSTALL_PREFIX}/setup_KWIVER.sh )

# ================================================================
# Setting up external projects for KWIVER components
#
include(ExternalProject)
set_property(DIRECTORY PROPERTY EP_STEP_TARGETS configure build install)

# Common CMake args to send to all external projects
set(KWIVER_ARGS_COMMON_CMAKE
  # CMAKE options
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX:PATH=${KWIVER_BUILD_INSTALL_PREFIX}
  -DBUILD_SHARED_LIBS:BOOL=${KWIVER_BUILD_SHARED}
  -DCMAKE_FIND_ROOT_PATH:PATH=${CMAKE_FIND_ROOT_PATH}
  -DCMAKE_LIBRARY_PATH:PATH=${CMAKE_LIBRARY_PATH}
  -DKWIVER_ENABLE_OPENCV:BOOL=${KWIVER_ENABLE_OPENCV}
  )

set(KWIVER_ARGS_COMMON_COMPILER
  # Compiler options
  -DCMAKE_CXX_COMPILER_ARG1:STRING=${CMAKE_CXX_COMPILER_ARG1}
  -DCMAKE_C_COMPILER_ARG1:STRING=${CMAKE_C_COMPILER_ARG1}
  -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
  -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
  -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
  -DCMAKE_CXX_FLAGS_MINSIZEREL:STRING=${CMAKE_CXX_FLAGS_MINSIZEREL}
  -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
  -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
  -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
  -DCMAKE_C_FLAGS_MINSIZEREL:STRING=${CMAKE_C_FLAGS_MINSIZEREL}
  -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
  -DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_C_FLAGS_RELWITHDEBINFO}
  )

set(KWIVER_ARGS_COMMON_BOOST
  # Boost files and options
  -DBOOST_ROOT=${BOOST_root}
  -DBoost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}
  -DBoost_LIBRARY_DIR=${Boost_LIBRARY_DIRS}
  -DBoost_USE_STATIC_LIBS=${Boost_USE_STATIC_LIBS}

  -DBoost_CHRONO_LIBRARY_DEBUG=${Boost_CHRONO_LIBRARY_DEBUG}
  -DBoost_CHRONO_LIBRARY_RELEASE=${Boost_CHRONO_LIBRARY_RELEASE}
  -DBoost_DATE_TIME_LIBRARY_DEBUG=${Boost_DATE_TIME_LIBRARY_DEBUG}
  -DBoost_DATE_TIME_LIBRARY_RELEASE=${Boost_DATE_TIME_LIBRARY_RELEASE}
  -DBoost_FILESYSTEM_LIBRARY_DEBUG=${Boost_FILESYSTEM_LIBRARY_DEBUG}
  -DBoost_FILESYSTEM_LIBRARY_RELEASE=${Boost_FILESYSTEM_LIBRARY_RELEASE}
  -DBoost_PROGRAM_OPTIONS_LIBRARY_DEBUG=${Boost_PROGRAM_OPTIONS_LIBRARY_DEBUG}
  -DBoost_PROGRAM_OPTIONS_LIBRARY_RELEASE=${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE}
  -DBoost_SYSTEM_LIBRARY_DEBUG=${Boost_SYSTEM_LIBRARY_DEBUG}
  -DBoost_SYSTEM_LIBRARY_RELEASE=${Boost_SYSTEM_LIBRARY_RELEASE}
  -DBoost_THREAD_LIBRARY_DEBUG=${Boost_THREAD_LIBRARY_DEBUG}
  -DBoost_THREAD_LIBRARY_RELEASE=${Boost_THREAD_LIBRARY_RELEASE}
)

set( KWIVER_ARGS_OPENCV )

if( KWIVER_ENABLE_OPENCV )
  set( KWIVER_ARGS_OPENCV
    -DOpenCV_DIR:PATH=${OpenCV_CONFIG_PATH}
    )
endif()

set(KWIVER_ARGS_COMMON_LINKER
  # Linker options
  "-DCMAKE_EXE_LINKER_FLAGS:STRING=${KWIVER_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}"
  "-DCMAKE_MODULE_LINKER_FLAGS:STRING=${KWIVER_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}"
  "-DCMAKE_SHARED_LINKER_FLAGS:STRING=${KWIVER_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}"
  )

set(KWIVER_ARGS_COMMON ${KWIVER_ARGS_COMMON_CMAKE}
                       ${KWIVER_ARGS_OPENCV}
                       ${KWIVER_ARGS_COMMON_COMPILER}
                       ${KWIVER_ARGS_COMMON_LINKER}
                       ${KWIVER_ARGS_COMMON_BOOST}
  )

# ================================================================
# Adding external components
#
include(External_VXL)
include(External_maptk)
include(External_sprokit)

# ================================================================
link_directories( ${KWIVER_BUILD_INSTALL_PREFIX}/lib )

add_subdirectory( kwiver )

# ================================================================
# build documentation
# - build doc for src
# - build doc for sprokit submodule
# - build doc for maptk submodule
# TBD
include(Doxygen)

# ================================================================
# Install rule if someone decided to local install
#
#install(
#  DIRECTORY ${KWIVER_BUILD_INSTALL_PREFIX}/
#  DESTINATION .
#  USE_SOURCE_PERMISSIONS
#  )


