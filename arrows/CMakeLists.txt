#
# Arrows Algorithms CMake file
#

set(ARROWS_SOURCE_DIR           "${CMAKE_CURRENT_SOURCE_DIR}")
set(ARROWS_BINARY_DIR           "${CMAKE_CURRENT_BINARY_DIR}")

# Base plug-in with default implementations
add_subdirectory( core )

# Add Ceres plugin if enabled
if( KWIVER_ENABLE_CERES )
  add_subdirectory( ceres )
endif()

# Add OpenCV plugin if enabled
if( KWIVER_ENABLE_OPENCV )
  add_subdirectory( ocv )
endif()

# Add PROJ plugin if enabled
if( KWIVER_ENABLE_PROJ )
  add_subdirectory( proj )
endif()

# Add VXL plugin if enabled
if( KWIVER_ENABLE_VXL )
  add_subdirectory( vxl )
endif()

# Add VisCL plugin if enabled
#  needs VXL
if( KWIVER_ENABLE_VISCL )
  add_subdirectory( viscl )
endif()

# if matlab is enabled
if( KWIVER_ENABLE_MATLAB )
  add_subdirectory( matlab )
endif()

# if darknet is enabled
if( KWIVER_ENABLE_DARKNET )
  add_subdirectory( darknet )
endif()

# if burnout is enabled
if( KWIVER_ENABLE_BURNOUT )
  add_subdirectory( burnout )
endif()

# if uuid is enabled
if( KWIVER_ENABLE_UUID )
  add_subdirectory( uuid )
endif()
