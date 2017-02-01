#
# Top level build for VITAL
#
project( vital )

###
# KWSys
#---------------------------------------------------------------------
# Create the kwsys library for vital.
set(KWSYS_NAMESPACE                kwiversys)
set(KWSYS_USE_DynamicLoader        1)
set(KWSYS_USE_SystemTools          1)
set(KWSYS_USE_System               1)
set(KWSYS_USE_RegularExpression    1)
set(KWSYS_USE_SystemInformation    1)
set(KWSYS_USE_Directory            1)
set(KWSYS_USE_CommandLineArguments 1)

set(KWSYS_INSTALL_EXPORT_NAME  ${kwiver_export_name})
set(KWSYS_INSTALL_INCLUDE_DIR  "include")
set(KWSYS_INSTALL_LIB_DIR      lib)
set(KWSYS_INCLUDE_DIR          ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

# Save our custom CXX flags and reset so we pass no special flags to kwiversys.
# Our flags are more strict than expected and causes many warnings and errors.
set( save_flags  ${CMAKE_CXX_FLAGS} )
set( CMAKE_CXX_FLAGS )

include_directories( SYSTEM ${KWSYS_INCLUDE_DIR}/kwiversys )
add_subdirectory(kwiversys)
_kwiver_export( ${KWSYS_NAMESPACE} )
set_property(GLOBAL APPEND PROPERTY kwiver_libraries kwiversys)

# force install to standard library directory
# Note that some of the following duplicates what is in kwiver_add_library()
set_target_properties( ${KWSYS_NAMESPACE}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  if (APPLE)
  set_target_properties( ${KWSYS_NAMESPACE}
    PROPERTIES
    MACOSX_RPATH          TRUE)
else()
  set_target_properties( ${KWSYS_NAMESPACE}
    PROPERTIES
        VERSION          ${KWIVER_VERSION}
        SOVERSION        ${KWIVER_VERSION}
        )
endif()

# restore old CXX flags for our code.
set( CMAKE_CXX_FLAGS ${save_flags} )

# ==================================================================
###
# Generate correct timer header based on availbility of timer
#

if (VITAL_USE_STD_CHRONO)

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/util/wall_timer.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/util/wall_timer.h" )

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/util/cpu_timer.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/util/cpu_timer.h" )

else()

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/util/no-wall_timer.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/util/wall_timer.h" )

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/util/no-cpu_timer.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/util/cpu_timer.h" )

endif()


# ==================================================================
###
# These headers are installed
#
set( vital_public_headers
  registrar.h
  vital_types.h
  vital_foreach.h
  noncopyable.h
  any.h
  algorithm_capabilities.h
  attribute_set.h

  exceptions.h
  exceptions/algorithm.h
  exceptions/base.h
  exceptions/image.h
  exceptions/io.h
  exceptions/math.h
  exceptions/video.h
  exceptions/klv.h
  exceptions/plugin.h

  io/camera_io.h
  io/camera_map_io.h
  io/eigen_io.h
  io/landmark_map_io.h
  io/mesh_io.h
  io/track_set_io.h

  types/bounding_box.h
  types/camera.h
  types/camera_intrinsics.h
  types/camera_map.h
  types/color.h
  types/covariance.h
  types/descriptor.h
  types/descriptor_set.h
  types/detected_object.h
  types/detected_object_set.h
  types/detected_object_type.h
  types/essential_matrix.h
  types/detected_object.h
  types/detected_object_set.h
  types/feature.h
  types/feature_set.h
  types/fundamental_matrix.h
  types/geo_MGRS.h
  types/geo_UTM.h
  types/geo_corner_points.h
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
  types/mesh.h
  types/polygon.h
  types/rotation.h
  types/similarity.h
  types/timestamp.h
  types/timestamp_config.h
  types/track.h
  types/track_set.h
  types/vector.h
  types/uuid.h

)

# ----------------------
set( vital_sources
  registrar.cxx
  algorithm_capabilities.cxx
  attribute_set.cxx

  exceptions/algorithm.cxx
  exceptions/base.cxx
  exceptions/image.cxx
  exceptions/io.cxx
  exceptions/math.cxx
  exceptions/video.cxx
  exceptions/klv.cxx
  exceptions/plugin.cxx

  io/camera_io.cxx
  io/camera_map_io.cxx
  io/landmark_map_io.cxx
  io/mesh_io.cxx
  io/track_set_io.cxx

  types/bounding_box.cxx
  types/camera.cxx
  types/camera_intrinsics.cxx
  types/detected_object.cxx
  types/detected_object_set.cxx
  types/detected_object_type.cxx
  types/essential_matrix.cxx
  types/feature.cxx
  types/fundamental_matrix.cxx
  types/geo_MGRS.cxx
  types/geo_UTM.cxx
  types/geo_corner_points.cxx
  types/geo_lat_lon.cxx
  types/homography.cxx
  types/homography_f2f.cxx
  types/homography_f2w.cxx
  types/image.cxx
  types/landmark.cxx
  types/mesh.cxx
  types/polygon.cxx
  types/rotation.cxx
  types/similarity.cxx
  types/timestamp.cxx
  types/track.cxx
  types/track_set.cxx
  types/uuid.cxx
)

kwiver_install_headers(
  ${vital_public_headers}
  SUBDIR   vital
  )

# install export header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_export.h
  ${CMAKE_CURRENT_BINARY_DIR}/vital_config.h
  SUBDIR   vital
  NOPATH
  )

# install generated header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/util/cpu_timer.h
  ${CMAKE_CURRENT_BINARY_DIR}/util/wall_timer.h
  SUBDIR   vital/util
  NOPATH
  )

kwiver_add_library( vital
  ${vital_public_headers}
  ${vital_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_export.h
  ${CMAKE_CURRENT_BINARY_DIR}/vital_config.h
  ${CMAKE_CURRENT_BINARY_DIR}/util/cpu_timer.h
  ${CMAKE_CURRENT_BINARY_DIR}/util/wall_timer.h
  )

target_link_libraries( vital
  PRIVATE         kwiversys
  PUBLIC          vital_config
                  vital_logger
                  vital_util
  )

###
# Build plugin management library
#
# This has to happen after plugin creation so as to be able to statically link
# in plugin libraries when building static.
#
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
  )

###
#
set(VITAL_MODULE_PATHS "${vital_default_module_paths}"
  CACHE STRING "The extra default paths for module scanning. Separate paths with ';' character.")
mark_as_advanced( VITAL_MODULE_PATHS )

# Configuring/Adding compile definitions to target
# (so we can use generator expressions)
if (WIN32)  # Windows puts DLLs in bin/CONFIG/
  set(shared_lib_dir "/bin")
  set(lib_dir_suffix)
  set(path_sep ";")
else()  # Other Unix systems
  set(shared_lib_dir "/lib")
  set(lib_dir_suffix "${LIB_SUFFIX}")
  set(path_sep ":")
endif()


if (CMAKE_CONFIGURATION_TYPES)
  set(config_subdir "/$<CONFIGURATION>")
endif()

set( install_lib_module_subdir "${shared_lib_dir}${lib_dir_suffix}/modules" )

#
# This is all to get around the ';' as the path separator.  CMake will
# split it into two elements when we want it to be ignored.
#

## The modules subdirectory is where the kwiver_add_module() macro generates its product
## Note that maptk has a different place where it installs plugins
set( install_module_paths  "${CMAKE_INSTALL_PREFIX}${install_lib_module_subdir}" )
set( build_module_paths    "${KWIVER_BINARY_DIR}${shared_lib_dir}${config_subdir}/modules" )

# assemble total search path
set( vital_default_module_path ${install_module_paths} )

# convert CMake list to path specification
foreach( p IN LISTS VITAL_MODULE_PATH )
  set( vital_default_module_path ${vital_default_module_path}${path_sep}${p} )
endforeach(p)

if (KWIVER_USE_BUILD_TREE)
  set( vital_default_module_path ${vital_default_module_path}${path_sep}${build_module_paths})
endif()

# used by APM
set( VITAL_DEFAULT_MODULE_PATH ${vital_default_module_path} CACHE STRING "Search path for loading algorithms" FORCE )

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/algorithm_plugin_manager_paths.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/algorithm_plugin_manager_paths.h" )

###
# configure our compiler options
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/vital_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/vital_config.h" )

###
#
add_subdirectory( algo )
add_subdirectory( util )
add_subdirectory( bindings )
add_subdirectory( config )
add_subdirectory( klv )
add_subdirectory( logger )
add_subdirectory( plugin_loader )
add_subdirectory( video_metadata )

if (KWIVER_ENABLE_TOOLS)
  add_subdirectory( tools )
endif()

###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
