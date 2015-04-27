#
# Top level build for kwiver pipelines
#

# ================================================================

include_directories( SYSTEM ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory( kwiver )

# other language bindings
add_subdirectory( bindings )
add_subdirectory( processes )

# kwiver_add_module(processes "${CMAKE_CURRENT_LIST_DIR}/processes")

#
# Run doxygen on all of kwiver
#
