#
# Build for vital exceptions
#

set( public_headers
  algorithm.h
  base.h
  image.h
  io.h
  math.h
  metadata.h
  video.h
  plugin.h
  serialize.h
)

set( sources
  algorithm.cxx
  base.cxx
  image.cxx
  io.cxx
  math.cxx
  metadata.cxx
  video.cxx
  plugin.cxx
  serialize.cxx
  )


# install export header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_exceptions_export.h
  SUBDIR   vital/exceptions
  NOPATH
  )

kwiver_install_headers(
  ${public_headers}
  SUBDIR vital/exceptions
  )

kwiver_add_library( vital_exceptions
  ${public_headers}
  ${sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_exceptions_export.h
  )
