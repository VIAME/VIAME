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
# Adding module subdirectories
#? kwiver_add_module( config "${CMAKE_CURRENT_LIST_DIR}/config" )
#?kwiver_add_module( logger "${CMAKE_CURRENT_LIST_DIR}/logger" )
add_subdirectory( config )
add_subdirectory( logger )

# add_subdirectory( processes )
