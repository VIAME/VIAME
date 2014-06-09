# kwiver/core

set( kwiver_core_public_headers
  exceptions.h
  timestamp.h
  geo_lat_lon.h
  config_util.h

  types/kwiver.h
  types/maptk.h
)

set( kwiver_core_sources
  timestamp.cxx
  geo_lat_lon.cxx
  config_util.cxx

  types/kwiver.cxx
)

add_library( kwiver_core
  ${kwiver_core_public_headers}
  ${kwiver_core_sources}
)
set_target_properties( kwiver_core PROPERTIES
  VERSION ${KWIVER_VERSION} SOVERSION ${KWIVER_VERSION}
  LIBRARY_OUTPUT_DIRECTORY ${KWIVER_BUILD_INSTALL_PREFIX}/lib
)

add_dependencies( kwiver_core sprokit maptk VXL )

target_link_libraries(kwiver_core
  ${Boost_LIBRARIES}
  )

# --- install headers ---
### TBD

install( TARGETS kwiver_core
  DESTINATION ${KWIVER_BUILD_INSTALL_PREFIX}/lib
)
