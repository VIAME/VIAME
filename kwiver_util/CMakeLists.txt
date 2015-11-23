#
# Make rules for kwiver-apps
#

project( kwiver_util )

include( kwiver-flags )

include_directories( ${CMAKE_CURRENT_LIST_DIR} )

add_subdirectory( processes )
add_subdirectory( bindings )
