#
# Build for vital exceptions
#

set( public_headers
  algorithm.h
  base.h
  gpu.h
  image.h
  io.h
  iteration.h
  math.h
  metadata.h
  video.h
  plugin.h
  serialize.h
)

set( sources
  algorithm.cxx
  base.cxx
  gpu.cxx
  image.cxx
  io.cxx
  iteration.cxx
  math.cxx
  metadata.cxx
  video.cxx
  plugin.cxx
  serialize.cxx
  )


# install export header

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_exceptions_export.h
  SUBDIR   viame/algorithm_framework/exceptions
  NOPATH
  )


viame_add_library( viame_exceptions
  ${public_headers}
  ${sources}
  ${CMAKE_CURRENT_BINARY_DIR}/viame_exceptions_export.h
  )

# Republished under the prefix the imported sources use.
viame_lite_generated( viame_exceptions_export.h algorithm_framework/exceptions )
