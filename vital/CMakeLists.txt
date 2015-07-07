#
# Top level build for VITAL
#
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fvisibility=hidden KWIVER_HAVE_GCC_VISIBILITY)

include( GenerateExportHeader )

set ( VITAL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set ( VITAL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} )

###
# Add the CMake directory for CMake modules
#
list(INSERT CMAKE_MODULE_PATH 0
  "${KWIVER_CORE_SOURCE_DIR}/CMake"
  )

# ==================================================================

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

###
#
set( vital_public_headers
  algorithm.h
  algorithm.txx
  camera.h
  camera_intrinsics.h
  camera_io.h
  camera_map.h
  camera_map_io.h
  covariance.h
  descriptor.h
  descriptor_set.h
  eigen_io.h
  exceptions.h
  feature.h
  feature_set.h
  geo_lat_lon.h
#  geo_reference_points_io.h
  homography_f2f.h
  homography_f2w.h
  homography.h
  image_container.h
  image.h
  ins_data.h
  ins_data_io.h
  landmark.h
  landmark_map.h
  landmark_map_io.h
#   local_geo_cs.h
  match_set.h
  matrix.h
  registrar.h
  rotation.h
  similarity.h
  timestamp.h
  track.h
  track_set.h
  track_set_io.h
  transform.h
  vector.h
  vital_config.h

  exceptions/algorithm.h
  exceptions/base.h
  exceptions/image.h
  exceptions/io.h
  exceptions/math.h
)

set( vital_sources
  algorithm.cxx
  camera.cxx
  camera_intrinsics.cxx
  camera_io.cxx
  camera_map_io.cxx
  feature.cxx
  geo_lat_lon.cxx
# geo_reference_points_io.cxx
  homography.cxx
  homography_f2f.cxx
  homography_f2w.cxx
  image.cxx
  ins_data.cxx
  ins_data_io.cxx
  landmark.cxx
  landmark_map_io.cxx
  #   local_geo_cs.cxx
  registrar.cxx
  rotation.cxx
  similarity.cxx
  timestamp.cxx
  track.cxx
  track_set.cxx
  track_set_io.cxx
  transform.cxx

  exceptions/algorithm.cxx
  exceptions/base.cxx
  exceptions/image.cxx
  exceptions/io.cxx
  exceptions/math.cxx
)

kwiver_install_headers(
  ${vital_public_headers}
  )

kwiver_add_library( vital SHARED
  ${vital_public_headers}
  ${vital_sources}
  )

target_link_libraries( vital
  PRIVATE       ${Boost_LIBRARIES} # be more specific
                kwiver_config
                kwiver_logger
  )

###
# Set options
set(KWIVER_LIB_SUFFIX "" CACHE STRING "String suffix appended to the library directory name we install into.")


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
set(vital_apm_headers_private
  algorithm_plugin_manager_static.h
  )
set(vital_apm_sources
  algorithm_plugin_manager.cxx
  algorithm_plugin_manager_static.cxx
  )

kwiver_install_headers(${vital_apm_headers})
kwiver_add_library(vital_apm
  ${vital_apm_headers}
  ${vital_apm_headers_private}
  ${vital_apm_sources}
  )

# Configuring/Adding compile definitions to target
# (so we can use generator expressions)
if (WIN32)  # Windows puts DLLs in bin/CONFIG/
  set(shared_lib_dir "/bin")
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

boolize(VITAL_USE_BUILD_PLUGIN_DIR)
boolize(VITAL_ENABLE_OPENCV)
boolize(VITAL_ENABLE_PROJ)
boolize(VITAL_ENABLE_VISCL)
boolize(VITAL_ENABLE_VXL)

target_compile_definitions(vital_apm
  PRIVATE
    # For plugin manager
    "SHARED_LIB_SUFFIX=\"${CMAKE_SHARED_MODULE_SUFFIX}\""
    "USE_BUILD_PLUGIN_DIR=${VITAL_USE_BUILD_PLUGIN_DIR}"
    "DEFAULT_PLUGIN_DIR_BUILD=\"${VITAL_BINARY_DIR}${shared_lib_dir}${config_subdir}/vital\""
    "DEFAULT_PLUGIN_DIR_INSTALL=\"${CMAKE_INSTALL_PREFIX}${shared_lib_dir}${lib_dir_suffix}/vital\""
    # For static interface
    # static interface support should be refactored back to maptk
    "VITAL_ENABLE_OPENCV=${VITAL_ENABLE_OPENCV}"
    "VITAL_ENABLE_PROJ=${VITAL_ENABLE_PROJ}"
    "VITAL_ENABLE_VISCL=${VITAL_ENABLE_VISCL}"
    "VITAL_ENABLE_VXL=${VITAL_ENABLE_VXL}"
  )

if (NOT WIN32)
  target_link_libraries(vital_apm ${CMAKE_DL_LIBS})
endif()

# When building static core, also link in static plugin libs
if(NOT BUILD_SHARED_LIBS)
  message(STATUS "Static build, linking plugin libraries into APM lib")
  get_property(plugin_static_libs
    GLOBAL PROPERTY vital_plugin_libraries
    )
  message(STATUS "Static build \"plugin\" libraries: ${plugin_static_libs}")
  target_link_libraries(vital_apm
    ${plugin_static_libs}
    )
endif()

###
#
add_subdirectory( bindings )

###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()

###
# Make appropriate export header file
generate_export_header( vital )
