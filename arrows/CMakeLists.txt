#
# Top level for arrows
#
if (KWIVER_ENABLE_CERES)
  option(KWIVER_ENABLE_ARROWS_CERES "Enable building arrows that depend on Ceres" ON )
else()
  unset(KWIVER_ENABLE_ARROWS_CERES CACHE)
endif()

if (KWIVER_ENABLE_VXL)
  option(KWIVER_ENABLE_ARROWS_VXL "Enable building arrows that depend on VXL" ON )
else()
  unset(KWIVER_ENABLE_ARROWS_VXL CACHE)
endif()

# option(KWIVER_ENABLE_ARROWS_OPENCV "Enable building arrows that depend on OPENCV" ON )
# option(KWIVER_ENABLE_ARROWS_PROJ "Enable building arrows that depend on Proj4" ON )
# option(KWIVER_ENABLE_ARROWS_VISCL "Enable building arrows that depend on Viscl" ON )
# option(KWIVER_ENABLE_ARROWS_CAFFE "Enable building arrows that depend on Caffe" ON )

add_subdirectory( algorithms )

if (ARROWS_ENABLE_SPROKIT)
  add_subdirectory( pipelines )
  add_subdirectory( processes )
endif()

set(arrows_DIR  ${CMAKE_CURRENT_BINARY_DIR}  CACHE PATH "Path to Arrows package")
#file(APPEND ${KWIVER_CONFIG_FILE} "set(arrows_DIR ${arrows_DIR})\n")
