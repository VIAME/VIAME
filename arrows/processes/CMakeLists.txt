#
#
#

add_subdirectory( bindings )
add_subdirectory( core )
add_subdirectory( examples )

if (ARROWS_ENABLE_VXL)
  add_subdirectory( vxl )
endif()

if (ARROWS_ENABLE_OPENCV)
  add_subdirectory( ocv )
endif()

if (ARROWS_ENABLE_PYTHON)
  add_subdirectory( python )
endif()
