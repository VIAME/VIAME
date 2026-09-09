# Configuration, logging, errors, the plugin registry and the algorithm
# interfaces
#
# Copied from the rest of kwiver's `vital` in P5-T02, unchanged; see the note
# in `core_types/CMakeLists.txt` for why the namespace and the include guards
# stay as they are.
#
# `core_types` compiles in here rather than into a library of its own: the two
# directories name each other, and CMake allows a cycle only between static
# libraries. See the note in `core_types/CMakeLists.txt`.

file( GLOB_RECURSE framework_headers CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.txx"
  )

file( GLOB_RECURSE framework_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
  )

# Where the plugin loader looks when nothing on the environment says
# otherwise, and where the pipeline parser looks for clusters and includes.
# Kwiver generated this; the values are this build's, so it cannot be a file
# in the source tree.
set( VITAL_MODULE_PATH
  "${VIAME_BUILD_INSTALL_PREFIX}/lib/kwiver/plugins" )
set( SPROKIT_DEFAULT_CLUSTER_PATHS
  "${VIAME_BUILD_INSTALL_PREFIX}/share/viame/config/pipelines/clusters" )
set( SPROKIT_DEFAULT_PIPE_INCLUDE_PATHS
  "${VIAME_BUILD_INSTALL_PREFIX}/share/viame/config/pipelines/include" )

if( WIN32 )
  set( path_sep "\;" )
else()
  set( path_sep ":" )
endif()

set( kwiver_vital_dir "${CMAKE_SOURCE_DIR}/packages/kwiver/vital" )
set( generated "${VIAME_GENERATED_INCLUDE_DIR}/viame/algorithm_framework" )

configure_file(
  "${kwiver_vital_dir}/vital_config.h.in"
  "${generated}/vital_config.h" )

configure_file(
  "${kwiver_vital_dir}/version.h.in"
  "${generated}/version.h" )

configure_file(
  "${kwiver_vital_dir}/kwiver-include-paths.h.in"
  "${generated}/kwiver-include-paths.h" )

kwiver_install_headers(
  SUBDIR     viame/algorithm_framework
  ${framework_headers}
  )

kwiver_add_library( viame_algorithm_framework
  ${framework_headers}
  ${framework_sources}
  ${VIAME_CORE_TYPES_HEADERS}
  ${VIAME_CORE_TYPES_SOURCES}
  "${generated}/vital_config.h"
  "${generated}/version.h"
  )

# The name the layout gives the types, so that a library depending on the
# types alone can say so and phase 8 can make it real.
add_library( viame_core_types ALIAS viame_algorithm_framework )

generate_export_header( viame_algorithm_framework
  BASE_NAME         VITAL_TYPES
  EXPORT_FILE_NAME  "${VIAME_GENERATED_INCLUDE_DIR}/viame/core_types/vital_types_export.h"
  )

kwiver_install_headers(
  "${VIAME_GENERATED_INCLUDE_DIR}/viame/core_types/vital_types_export.h"
  NOPATH   SUBDIR   viame/core_types
  )

# One export header per macro name the copied sources use. Kwiver had a
# library for each; the macros all resolve to default visibility here, so one
# library can answer to all of them.
foreach( pair
    "VITAL:vital_export.h"
    "VITAL_CONFIG:config/vital_config_export.h"
    "VITAL_EXCEPTIONS:exceptions/vital_exceptions_export.h"
    "VITAL_LOGGER:logger/vital_logger_export.h"
    "VITAL_UTIL:util/vital_util_export.h"
    "VITAL_ALGO:algo/vital_algo_export.h"
    "VITAL_APPLETS:applets/vital_applets_export.h"
    "VITAL_VPM:plugin/vital_vpm_export.h" )

  string( REPLACE ":" ";" pair "${pair}" )
  list( GET pair 0 base_name )
  list( GET pair 1 header )

  generate_export_header( viame_algorithm_framework
    BASE_NAME         ${base_name}
    EXPORT_FILE_NAME  "${generated}/${header}"
    )

  get_filename_component( header_dir "${header}" DIRECTORY )

  kwiver_install_headers(
    "${generated}/${header}"
    NOPATH   SUBDIR   viame/algorithm_framework/${header_dir}
    )
endforeach()

kwiver_install_headers(
  "${generated}/vital_config.h"
  "${generated}/version.h"
  NOPATH   SUBDIR   viame/algorithm_framework
  )

target_include_directories( viame_algorithm_framework PUBLIC
  $<BUILD_INTERFACE:${VIAME_SOURCE_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${VIAME_GENERATED_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cxxopts>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cereal>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/rapidjson>
  $<INSTALL_INTERFACE:include>
  )

target_link_libraries( viame_algorithm_framework
  PUBLIC               Eigen3::Eigen
  PRIVATE              kwiver::kwiversys
                       ${CMAKE_DL_LIBS}
  )

set_target_properties( viame_algorithm_framework PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )
