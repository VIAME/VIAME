#
# Top level build for kwiver pipelines
#

# ================================================================

include_directories( SYSTEM ${CMAKE_CURRENT_SOURCE_DIR} )

###
# KWSys borrowed from CMake
#---------------------------------------------------------------------
# Create the kwsys library for kwiver.
set(KWSYS_NAMESPACE kwiversys)
set(KWSYS_USE_DynamicLoader 1)
set(KWSYS_INSTALL_EXPORT_NAME kwiver)
set(KWSYS_INSTALL_LIB_DIR lib)
set(KWSYS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/kwiversys CACHE INTERNAL "")

include_directories( SYSTEM ${KWSYS_INCLUDE_DIR} )
add_subdirectory(kwiversys)
## export( TARGETS kwiversys APPEND FILE "${KWIVER_EXPORT_FILE}" )

###
#
add_subdirectory( kwiver )

# other language bindings
add_subdirectory( bindings )
add_subdirectory( processes )

# kwiver_add_module(processes "${CMAKE_CURRENT_LIST_DIR}/processes")

#
# Run doxygen on all of kwiver
#
