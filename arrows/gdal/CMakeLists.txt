# Build / Install plugin containing GDAL algorithm implementations + support
# structures

set( gdal_headers_public
  image_container.h
  image_io.h
  )

kwiver_install_headers(
  SUBDIR     arrows/gdal
  ${gdal_headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_gdal_export.h
  NOPATH   SUBDIR     arrows/gdal
  )

set( gdal_sources
  image_container.cxx
  image_io.cxx
  )

kwiver_add_library( kwiver_algo_gdal
  ${gdal_headers_public}
  ${gdal_sources}
  )
target_link_libraries( kwiver_algo_gdal
  PUBLIC               kwiver_algo_core
  PRIVATE              ${GDAL_LIBRARY}
  )

algorithms_create_plugin( kwiver_algo_gdal
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
