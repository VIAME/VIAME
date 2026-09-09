# In-house image kernels
#
# Header only and templated over the pixel type, so there is nothing to link
# against: a consumer includes what it needs and takes vital::image with it.
# Nothing here may include OpenCV or VXL.

set( image_ops_headers
  channels.h
  commonality.h
  convert.h
  dispatch.h
  morphology.h
  pixel.h
  statistics.h
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
