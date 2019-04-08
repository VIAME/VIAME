###
# Build vital/util library
#
# A hard requirement for the vital_util library is that it not depend
# on vital library.
#

# ==================================================================
###
# These headers are installed
#
set( public_headers
  get_paths.h
  timer.h
  cpu_timer.h
  wall_timer.h
  demangle.h
  any_converter.h
  enumerate_matrix.h
  buffer.h
  ring_buffer.h
  bounded_buffer.h
  source_location.h
  data_stream_reader.h
  hex_dump.h
  string.h
  string_editor.h
  simple_stats.h
  thread_pool.h
  token_expander.h
  token_expand_editor.h
  token_type_env.h
  token_type.h
  token_type_symtab.h
  token_type_sysenv.h
  tokenize.h
  transform_image.h
  wrap_text_block.h
  file_md5.h
  )

# ----------------------
set( private_headers
  thread_pool_builtin_backend.h
  thread_pool_gcd_backend.h
  thread_pool_sync_backend.h
  )

# ----------------------
set( sources
  get_paths.cxx
  demangle.cxx
  source_location.cxx

  data_stream_reader.cxx
  hex_dump.cxx
  string.cxx
  string_editor.cxx
  thread_pool.cxx
  token_expander.cxx
  token_expand_editor.cxx
  token_type.cxx
  token_type_env.cxx
  token_type_symtab.cxx
  token_type_sysenv.cxx
  wrap_text_block.cxx
  file_md5.cxx
  )

kwiver_install_headers(
  ${public_headers}
  SUBDIR   vital/util
  )

# install generated header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_util_export.h
  SUBDIR   vital/util
  NOPATH
  )

kwiver_add_library( vital_util
  ${public_headers}
  ${private_headers}
  ${sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_util_export.h
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
