#
#
#

add_subdirectory( bindings )
add_subdirectory( core )

if (KWIVER_ENABLE_VXL)
  add_subdirectory( vxl )
endif()

if (KWIVER_ENABLE_OPENCV)
  add_subdirectory( ocv )
endif()

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( python )
endif()
