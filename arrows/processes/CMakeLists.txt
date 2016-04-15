#
# Arrows processes
#

option(ARROWS_ENABLE_PROCESS_C_BINDINGS "Enable building C bindings for processes" OFF )
option(ARROWS_ENABLE_PROCESS_EXAMPLES "Enable building process examples" ON )

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

if (ARROWS_ENABLE_CAFFE)
#  add_subdirectory(caffe)
endif()
