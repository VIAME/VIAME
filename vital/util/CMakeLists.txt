###
# Build vital/util library
#
# A hard requirement for the vital_util library is that it not depend
# on vital library.
#

# ==================================================================
###
# Generate correct timer header based on availbility of timer
#

if (VITAL_USE_STD_CHRONO)

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/wall_timer.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/wall_timer.h" )

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu_timer.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cpu_timer.h" )

else()

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/no-wall_timer.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/wall_timer.h" )

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/no-cpu_timer.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cpu_timer.h" )

endif()

# ==================================================================
###
# These headers are installed
#
set( public_headers
  get_paths.h
  timer.h
  demangle.h
  any_converter.h
  enumerate_matrix.h
  bounded_buffer.h
  source_location.h

  data_stream_reader.h
  string_editor.h
  token_expander.h
  token_expand_editor.h
  token_type_env.h
  token_type.h
  token_type_symtab.h
  token_type_sysenv.h
  transform_image.h
  wrap_text_block.h
  )

# ----------------------
set( sources
  get_paths.cxx
  demangle.cxx
  source_location.cxx

  data_stream_reader.cxx
  string_editor.cxx
  token_expander.cxx
  token_expand_editor.cxx
  token_type.cxx
  token_type_env.cxx
  token_type_symtab.cxx
  token_type_sysenv.cxx
  wrap_text_block.cxx
  )

kwiver_install_headers(
  ${public_headers}
  SUBDIR   vital/util
  )

# install generated header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_util_export.h
  ${CMAKE_CURRENT_BINARY_DIR}/cpu_timer.h
  ${CMAKE_CURRENT_BINARY_DIR}/wall_timer.h
  SUBDIR   vital/util
  NOPATH
  )

kwiver_add_library( vital_util
  ${public_headers}
  ${sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_util_export.h
  ${CMAKE_CURRENT_BINARY_DIR}/cpu_timer.h
  ${CMAKE_CURRENT_BINARY_DIR}/wall_timer.h
  )

target_link_libraries( vital_util
  PRIVATE         kwiversys
                  vital_logger
  )

###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
