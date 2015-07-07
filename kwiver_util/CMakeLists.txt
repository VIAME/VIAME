#
# Make rules for kwiver-apps
#

project( kwiver_util )


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

###
#
add_subdirectory( config )
add_subdirectory( logger )



# add_subdirectory( processes )

###
#
_kwiver_export( kwiversys )
