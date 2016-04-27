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
set(KWSYS_INSTALL_INCLUDE_DIR  "include/kwiver")
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

  algo/algorithm.txx
  algo/algorithm.h
  algo/analyze_tracks.h
  algo/bundle_adjust.h
  algo/close_loops.h
  algo/compute_ref_homography.h
  algo/convert_image.h
  algo/detect_features.h
  algo/draw_tracks.h
  algo/estimate_essential_matrix.h
  algo/estimate_fundamental_matrix.h
  algo/estimate_homography.h
  algo/estimate_similarity_transform.h
  algo/extract_descriptors.h
  algo/filter_features.h
  algo/geo_map.h
  algo/image_io.h
  algo/initialize_cameras_landmarks.h
  algo/match_features.h
  algo/optimize_cameras.h
  algo/track_features.h
  algo/triangulate_landmarks.h

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
  types/color.h
  types/covariance.h
  types/descriptor.h
  types/descriptor_set.h
  types/essential_matrix.h
  types/feature.h
  types/feature_set.h
  types/fundamental_matrix.h
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
  types/timestamp_config.h
  types/track.h
  types/track_set.h
  types/vector.h

  util/get_paths.h
  util/timer.h
  util/enumerate_matrix.h
)

# ----------------------
set( vital_sources
  registrar.cxx

  algo/algorithm.cxx
  algo/analyze_tracks.cxx
  algo/bundle_adjust.cxx
  algo/close_loops.cxx
  algo/compute_ref_homography.cxx
  algo/convert_image.cxx
  algo/detect_features.cxx
  algo/draw_tracks.cxx
  algo/estimate_essential_matrix.cxx
  algo/estimate_fundamental_matrix.cxx
  algo/estimate_homography.cxx
  algo/estimate_similarity_transform.cxx
  algo/extract_descriptors.cxx
  algo/filter_features.cxx
  algo/geo_map.cxx
  algo/image_io.cxx
  algo/initialize_cameras_landmarks.cxx
  algo/match_features.cxx
  algo/optimize_cameras.cxx
  algo/track_features.cxx
  algo/triangulate_landmarks.cxx

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
  types/essential_matrix.cxx
  types/feature.cxx
  types/fundamental_matrix.cxx
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

  util/get_paths.cxx
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
  )

###
# Build plugin management library
#
# This has to happen after plugin creation so as to be able to statically link
# in plugin libraries when building static.
#
option(VITAL_USE_BUILD_PLUGIN_DIR
  "Whether to include the build-tree plugin directory to the default plugin path list"
  OFF
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
  )

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

set( install_lib_module_subdir "${shared_lib_dir}${lib_dir_suffix}/arrows" )

#
# This is all to get around the ';' as the path separator.  CMake will
# split it into two elements when we want it to be ignored.
#

## The modules subdirectory is where the kwiver_add_module() macro generates its product
## Note that maptk has a different place where it installs plugins
set( install_module_paths  "${CMAKE_INSTALL_PREFIX}${install_lib_module_subdir}" )
set( build_module_paths    "${KWIVER_BINARY_DIR}${shared_lib_dir}${config_subdir}/arrows" )

# assemble total search path
set( VITAL_DEFAULT_MODULE_PATHS ${install_module_paths})

# convert CMake list to path specification
foreach( p IN LISTS VITAL_MODULE_PATHS )
  set( VITAL_DEFAULT_MODULE_PATHS ${VITAL_DEFAULT_MODULE_PATHS}${path_sep}${p} )
endforeach(p)

if (VITAL_USE_BUILD_PLUGIN_DIR)
  set( VITAL_DEFAULT_MODULE_PATHS ${VITAL_DEFAULT_MODULE_PATHS}${path_sep}${build_module_paths})
endif()

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
add_subdirectory( bindings )
add_subdirectory( logger )
add_subdirectory( config )

option (VITAL_ENABLE_TOOLS "Enable tools" ON )
if (VITAL_ENABLE_TOOLS)
  add_subdirectory( tools )
endif()

###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
  add_subdirectory( util/tests )
endif()

set(vital_DIR  ${CMAKE_CURRENT_BINARY_DIR}  CACHE PATH "Path to Vital package")
# file(APPEND ${KWIVER_CONFIG_FILE} "set(vital_DIR ${vital_DIR})\n")

###
# Marking source root for documentation generation
#
kwiver_create_doxygen( vital "${CMAKE_CURRENT_LIST_DIR}" )
