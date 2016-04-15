#
#
#

option(KWIVER_ENABLE_PROCESS_C_BINDINGS "Enable building C bindings for processes" OFF )
option(KWIVER_ENABLE_PROCESS_EXAMPLES "Enable building process examples" ON )

add_subdirectory( bindings )
add_subdirectory( core )
add_subdirectory( examples )

if (KWIVER_ENABLE_VXL)
  add_subdirectory( vxl )
endif()

if (KWIVER_ENABLE_OPENCV)
  add_subdirectory( ocv )
endif()

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( python )
endif()
