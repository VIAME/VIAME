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

include_directories( ${CMAKE_CURRENT_BINARY_DIR} ) # for generated files

kwiver_install_headers(
  ${public_headers}
  SUBDIR     processes/bindings/c
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_type_converters_export.h
  SUBDIR      processes/bindings/c
  NOPATH
  )

kwiver_add_library( vital_type_converters
  ${public_headers}
  ${private_headers}
  ${sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_type_converters_export.h
  )

target_link_libraries( vital_type_converters
  PRIVATE              vital
                       vital_python_util
                       sprokit_pipeline
                       vital_c
                       vital_logger
                     ${PYTHON_LIBRARIES}
  )
