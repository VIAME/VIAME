#
# C interface library for Kwiver access
#

set( public_headers
  vital_type_converters.h
  )

set( private_headers
  )


set( sources
  vital_type_converters.cxx
  )

kwiver_install_headers(
  ${public_headers}
  SUBDIR     kwiver_util/bindings/c
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_type_converters_export.h
  SUBDIR      kwiver_util/bindings/c
  NOPATH
  )

kwiver_add_library( vital_type_converter
  ${public_headers}
  ${private_headers}
  ${sources}
  )

target_link_libraries( vital_type_converter
  PRIVATE              vital
                       sprokit_pipeline
                       vital_c
                       vital_logger
                       ${PYTHON_LIBRARIES}
  )
