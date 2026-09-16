# The algorithm framework, and the types it is written in terms of
#
# Vital, imported from kwiver in P5-T02 and built from here since P5-T05.
# `core_types` is a subdirectory of this one in the build even though it is a
# sibling in the tree: the two name each other -- the types need
# configuration, logging and errors, and the algorithm interfaces are written
# in terms of the types -- and CMake allows a cycle only between static
# libraries. Kwiver cut the same code into ten libraries to keep the graph
# acyclic and this keeps that cut; which of them are worth having is phase 8's
# question.
project( vital )
set(CMAKE_FOLDER "Vital")

# ==============================================================================
# Other VITAL-specific setup and flags

# ==============================================================================
# Vital Library

# ==================================================================
###
# These headers are installed
#
set( vital_core_types_headers
  ${CMAKE_CURRENT_SOURCE_DIR}/../core_types/vital_types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../core_types/any.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../core_types/attribute_set.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../core_types/bitflags.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../core_types/context.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../core_types/iterator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../core_types/math_constants.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../core_types/noncopyable.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../core_types/set.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../core_types/signal.h
)

set( vital_framework_headers
  algorithm_capabilities.h
  exceptions.h
  overload.h

  io/camera_from_metadata.h
  io/eigen_io.h
  io/mesh_io.h
  io/metadata_io.h
)

set( vital_sources
  algorithm_capabilities.cxx

  io/camera_from_metadata.cxx
  io/mesh_io.cxx
  io/metadata_io.cxx
)

# The imported tree already has the shape it should install with, so it is
# installed as a tree. viame_install_headers turns an absolute path into a
# relative one under the destination, which for these would put the whole
# build machine's path under include/viame/.
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../core_types/"
  DESTINATION "include/viame/core_types"
  FILES_MATCHING PATTERN "*.h" PATTERN "*.txx"
  PATTERN "CMakeLists.txt" EXCLUDE
  # the unit tests beside the code since P2-T09; FILES_MATCHING would still
  # create the empty directory
  PATTERN "tests" EXCLUDE )

install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION "include/viame/algorithm_framework"
  FILES_MATCHING PATTERN "*.h" PATTERN "*.txx"
  PATTERN "CMakeLists.txt" EXCLUDE
  PATTERN "tests" EXCLUDE )

###
# configure our compiler options
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/vital_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/vital_config.h" )

viame_add_library( vital
  ${vital_core_types_headers}
  ${vital_framework_headers}
  ${vital_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_export.h
  ${CMAKE_CURRENT_BINARY_DIR}/vital_config.h
  )

viame_lite_generated( vital_export.h  algorithm_framework )
viame_lite_generated( vital_config.h  algorithm_framework )
# version.h is republished from the top-level CMakeLists: `kwiver_configure_file`
# writes it with the `viame_configure` target at build time, so there is
# nothing here to copy at configure time.

# Make sure the configuration targets flush before building this library.
add_dependencies (vital viame_configure)

target_link_libraries( vital
  PRIVATE
  PUBLIC          vital_config
                  vital_exceptions
                  vital_logger
                  vital_util
                  vital_types
                  vital_vpm
                  vital_applets
  )

###
#
# This option adds another directory to the path by appending the configuration type to
# existing directories.
set(default OFF)

if (CMAKE_CONFIGURATION_TYPES)
  set(default ON)
endif ()

option(KWIVER_USE_CONFIGURATION_SUBDIRECTORY
  "Look in the configuration's subdirectory for each module path (e.g. debug, release, ...)" ${default})

###
# Subdirectory entries
# -- Creates library "vital_exceptions"
add_subdirectory( exceptions )
add_subdirectory( algo )
# -- Creates library "vital_util"
add_subdirectory( util )
add_subdirectory( range )
# -- Creates library "vital_config"
# Requires: vital_logger, vital_util (vital_types)
add_subdirectory( config )
# -- Creates library "vital_logger"
# Requires: vital_util
add_subdirectory( logger )
# -- Creates library "vital_types"
# Requires: vital_config
add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/../core_types" core_types )
# -- Creates library "vital_vpm"
# Requires: vital_exceptions, vital_logger, vital_util
# Uses common headers: vital_types.h
add_subdirectory( plugin )
# -- Created library "vital_applets"
add_subdirectory( applets )

# ==============================================================================
# The imported vital's include roots, on the targets rather than on the
# directory, so that they reach anything that links these -- kwiver's own
# arrows and sprokit, and VIAME, which consumes them through the export set.
foreach( _viame_lite_target
    vital vital_types vital_config vital_logger vital_exceptions
    vital_util vital_vpm vital_algo vital_applets )

  if( TARGET ${_viame_lite_target} )
    # The namespaced name VIAME's CMakeLists use. It came from kwiver's
    # installed config package until P1-T05 and from an alias file until
    # P5-T05 emptied that of everything but this loop.
    if( NOT TARGET kwiver::${_viame_lite_target} )
      add_library( kwiver::${_viame_lite_target} ALIAS ${_viame_lite_target} )
    endif()

    target_include_directories( ${_viame_lite_target} PUBLIC
      $<BUILD_INTERFACE:${VIAME_LITE_SOURCE_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${VIAME_LITE_GENERATED_DIR}>
      $<BUILD_INTERFACE:${VIAME_LITE_ROOT_DIR}/library/tpl/rapidjson>
      $<BUILD_INTERFACE:${VIAME_LITE_ROOT_DIR}/library/tpl/cxxopts>
      $<INSTALL_INTERFACE:include>
      )
  endif()
endforeach()

# ==============================================================================
# Built-in plugin modules
#
# This is specifically after the Vital library definition as some of these
# require the vital library for parts and utility.
# And, you know, the plugin management stuff (except for the logger, which works
# independently from the main plugin management so that it can have logging
# itself).
#
# The sources are VIAME's now, as the library itself is: P5-T05 moved them to
# library/algorithm_framework.
#
# `logger_plugins` went instead of moving -- both of its plugins are behind
# KWIVER_ENABLE_LOG4CXX and KWIVER_ENABLE_LOG4CPLUS, and neither has been on
# in this build; so did `tools`, which upstream had commented out in its
# entirety, `types/protobuf`, and this directory's own tests.
add_subdirectory( config_plugins )
add_subdirectory( applets_plugins )
add_subdirectory( test_interface )

###
# Python bindings for the algorithm interfaces
#
# `kwiver.vital.algo`, moved here from `python/kwiver/vital/algo` by P8-T02
# along with the deletion of the castxml generator that used to write them.
##
if( VIAME_ENABLE_PYTHON )
  include( "${CMAKE_CURRENT_SOURCE_DIR}/algo/python.cmake" )
endif()

###
# Tests
##
if( VIAME_ENABLE_TESTS )
  add_subdirectory( tests )
endif()
