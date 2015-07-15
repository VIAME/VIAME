#
# Top level build for VITAL
#
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fvisibility=hidden KWIVER_HAVE_GCC_VISIBILITY)

include( GenerateExportHeader )

###
# KWSys borrowed from CMake
#---------------------------------------------------------------------
# Create the kwsys library for vital.
set(KWSYS_NAMESPACE kwiversys)
set(KWSYS_USE_DynamicLoader 1)
set(KWSYS_USE_SystemTools 1)
set(KWSYS_USE_RegularExpression 1)
set(KWSYS_INSTALL_EXPORT_NAME  ${kwiver_export_name})
set(KWSYS_INSTALL_LIB_DIR lib)
set(KWSYS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

include_directories( SYSTEM ${KWSYS_INCLUDE_DIR}/kwiversys )
add_subdirectory(kwiversys)
_kwiver_export( ${KWSYS_NAMESPACE} )

# ==================================================================

###
#
set( vital_public_headers
  registrar.h
  vital_config.h

  algo/algorithm.txx
  algo/algorithm.h
  algo/convert_image.h
  algo/image_io.h
  algo/track_features.h

  exceptions.h
  exceptions/algorithm.h
  exceptions/base.h
  exceptions/image.h
  exceptions/io.h
  exceptions/math.h

  io/camera_io.h
  io/camera_map_io.h
  io/eigen_io.h
  io/landmark_map_io.h
  io/track_set_io.h

  types/camera.h
  types/camera_intrinsics.h
  types/camera_map.h
  types/covariance.h
  types/descriptor.h
  types/descriptor_set.h
  types/feature.h
  types/feature_set.h
  types/geo_lat_lon.h
  types/homography.h
  types/homography_f2f.h
  types/homography_f2w.h
  types/image.h
  types/image_container.h
  types/landmark.h
  types/landmark_map.h
  types/match_set.h
  types/matrix.h
  types/rotation.h
  types/similarity.h
  types/timestamp.h
  types/track.h
  types/track_set.h
  types/transform.h
  types/vector.h
)

set( vital_sources
  registrar.cxx

  algo/algorithm.cxx
  algo/convert_image.cxx
  algo/image_io.cxx
  algo/track_features.cxx

  exceptions/algorithm.cxx
  exceptions/base.cxx
  exceptions/image.cxx
  exceptions/io.cxx
  exceptions/math.cxx

  io/camera_io.cxx
  io/camera_map_io.cxx
  io/landmark_map_io.cxx
  io/track_set_io.cxx

  types/camera.cxx
  types/camera_intrinsics.cxx
  types/feature.cxx
  types/geo_lat_lon.cxx
  types/homography.cxx
  types/homography_f2f.cxx
  types/homography_f2w.cxx
  types/image.cxx
  types/landmark.cxx
  types/rotation.cxx
  types/similarity.cxx
  types/timestamp.cxx
  types/track.cxx
  types/track_set.cxx
  types/transform.cxx
)

kwiver_install_headers(
  ${vital_public_headers}
  SUBDIR   vital
  )

# install export header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_export.h
  SUBDIR   vital
  NOPATH
  )

kwiver_add_library( vital
  ${vital_public_headers}
  ${vital_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_export.h
  )

target_link_libraries( vital
  PRIVATE         ${Boost_LIBRARIES} # be more specific
                  kwiversys
  PUBLIC          vital_config
                  vital_logger
  )


###
# Build plugin management library
#
# This has to happen after plugin creation so as to be able to statically link
# in plugin libraries when building static.
#
option(KWIVER_USE_BUILD_PLUGIN_DIR
  "Whether to include the build-tree plugin directory to the default plugin path list"
  ON
  )

set(vital_apm_headers
  algorithm_plugin_manager.h
  )
set(vital_apm_sources
  algorithm_plugin_manager.cxx
  )
set(vital_apm_headers_private
  ${CMAKE_CURRENT_BINARY_DIR}/vital_apm_export.h
  )

kwiver_add_library(vital_apm
  ${vital_apm_headers}
  ${vital_apm_headers_private}
  ${vital_apm_sources}
  )

kwiver_install_headers(
  ${vital_apm_headers}
  SUBDIR      vital
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_apm_export.h
  SUBDIR      vital
  NOPATH
  )

target_link_libraries(vital_apm
  PRIVATE          kwiversys
                   vital
                   vital_logger
                   boost_filesystem
  )

# Configuring/Adding compile definitions to target
# (so we can use generator expressions)
if (WIN32)  # Windows puts DLLs in bin/CONFIG/
  set(shared_lib_dir "/bin")
  set(lib_dir_suffix)
else()  # Other Unix systems
  set(shared_lib_dir "/lib")
  set(lib_dir_suffix "${KWIVER_LIB_SUFFIX}")
endif()

if (CMAKE_CONFIGURATION_TYPES)
  set(config_subdir "/$<CONFIGURATION>")
endif()

# Convert use-build-dir switch into 1/0 value
macro(boolize var_name)
  if( ${var_name} )
    set(${var_name} 1)
  else()
    set(${var_name} 0)
  endif()
endmacro()

boolize(KWIVER_USE_BUILD_PLUGIN_DIR)

# plugin manager needs a lot of external info so it can locate plugins
# and such
target_compile_definitions(vital_apm
  PRIVATE
    "SHARED_LIB_SUFFIX=\"${CMAKE_SHARED_MODULE_SUFFIX}\""
    "USE_BUILD_PLUGIN_DIR=${KWIVER_USE_BUILD_PLUGIN_DIR}"
    "DEFAULT_PLUGIN_DIR_BUILD=\"${KWIVER_BINARY_DIR}${shared_lib_dir}${config_subdir}/modules\""
    "DEFAULT_PLUGIN_DIR_INSTALL=\"${CMAKE_INSTALL_PREFIX}${shared_lib_dir}${lib_dir_suffix}/modules\""
  )

###
#
add_subdirectory( bindings )
add_subdirectory( logger )
add_subdirectory( config )
add_subdirectory( tools )

###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()

###
# Make appropriate export header file
generate_export_header( vital )
