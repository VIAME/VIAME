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
  add_subdirectory( dbow2 )
  add_subdirectory( ocv )
endif()

# Add PROJ plugin if enabled
if( KWIVER_ENABLE_PROJ )
  add_subdirectory( proj )
endif()

# Add Qt plugin if enabled
if( KWIVER_ENABLE_QT )
  add_subdirectory( qt )
endif()

# Add VXL plugin if enabled
if( KWIVER_ENABLE_VXL )
  add_subdirectory( vxl )

  # Super3D depends on VXL
  option( KWIVER_ENABLE_SUPER3D
    "Enable SUPER3D dependent code and plugins (Arrows)"
    ON
    )

  # Add super3d plugin if enabled
  if( KWIVER_ENABLE_SUPER3D )
    add_subdirectory( super3d )
  endif()
endif()

if ( KWIVER_ENABLE_CUDA )
  add_subdirectory( cuda )
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

# if kpf is enabled
if( KWIVER_ENABLE_KPF )
  add_subdirectory( kpf )
endif()

# if GDAL is enabled
if ( KWIVER_ENABLE_GDAL )
  add_subdirectory( gdal )
endif()

# if uuid is enabled
if( KWIVER_ENABLE_UUID )
  add_subdirectory( uuid )
endif()


# if FFMPEG is enabled
if( KWIVER_ENABLE_FFMPEG)
  add_subdirectory( ffmpeg )
endif()

add_subdirectory( serialize )
