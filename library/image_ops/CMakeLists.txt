# In-house image kernels
#
# Header only and templated over the pixel type, so there is nothing to link
# against: a consumer includes what it needs and takes vital::image with it.
# Nothing here may include OpenCV or VXL.

set( image_ops_headers
  channels.h
  color.h
  commonality.h
  convert.h
  dispatch.h
  homography_overlap.h
  morphology.h
  pixel.h
  polygon.h
  resample.h
  statistics.h
  stretch.h
  temporal.h
  threshold.h
)

add_library( viame_image_ops INTERFACE )
add_library( viame::image_ops ALIAS viame_image_ops )

# The parent directory, so consumers include <image_ops/convert.h> rather than
# <convert.h>: several kernels share a name with the algorithm that uses them
target_include_directories( viame_image_ops INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:include/viame>
)

target_link_libraries( viame_image_ops INTERFACE kwiver::vital )

install( FILES       ${image_ops_headers}
         DESTINATION include/viame/image_ops )

# In the export set, because the libraries that link it are: since P5-T05
# VIAME installs a config package of its own and every target another one
# names has to be exportable.
install( TARGETS     viame_image_ops
         EXPORT      ${kwiver_export_name} )
set_property( GLOBAL APPEND PROPERTY kwiver_export_targets viame_image_ops )
