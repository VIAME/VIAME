#
# Top level for arrows
#
project( arrows )

CMAKE_DEPENDENT_OPTION(ARROWS_ENABLE_CERES "Enable building arrows that depend on Ceres" ON
  KWIVER_ENABLE_CERES OFF )

CMAKE_DEPENDENT_OPTION(ARROWS_ENABLE_VXL "Enable building arrows that depend on VXL" ON
  KWIVER_ENABLE_VXL OFF)

CMAKE_DEPENDENT_OPTION(ARROWS_ENABLE_OPENCV "Enable building arrows that depend on OPENCV" ON
  KWIVER_ENABLE_OPENCV OFF)

CMAKE_DEPENDENT_OPTION(ARROWS_ENABLE_PROJ "Enable building arrows that depend on Proj4" ON
  KWIVER_ENABLE_PROJ OFF)

CMAKE_DEPENDENT_OPTION(ARROWS_ENABLE_VISCL "Enable building arrows that depend on Viscl" ON
  KWIVER_ENABLE_VISCL OFF)

option(ARROWS_ENABLE_CAFFE "Enable building arrows that depend on Caffe" ON)

add_subdirectory( algorithms )

if (KWIVER_ENABLE_SPROKIT)
  # TODO add include_directories( sprokit )
  add_subdirectory( pipelines )
  add_subdirectory( processes )
endif()

set(arrows_DIR  ${CMAKE_CURRENT_BINARY_DIR}  CACHE PATH "Path to Arrows package")
#file(APPEND ${KWIVER_CONFIG_FILE} "set(arrows_DIR ${arrows_DIR})\n")
