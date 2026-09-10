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
  bounded_buffer.h
  cpu_timer.h
  data_stream_reader.h
  demangle.h
  directed_graph.h
  get_paths.h
  rational.h
  source_location.h
  streamable.h
  string.h
  string_editor.h
  timer.h
  token_expander.h
  token_type.h
  token_type_env.h
  token_type_symtab.h
  token_type_sysenv.h
  tokenize.h
  transform_image.h
  visit.h
  wall_timer.h
  wrap_text_block.h
  )

# ----------------------
set( private_headers
  )

# ----------------------
set( sources
  get_paths.cxx
  demangle.cxx
  source_location.cxx

  data_stream_reader.cxx
  string.cxx
  string_editor.cxx
  token_expander.cxx
  token_type.cxx
  token_type_env.cxx
  token_type_symtab.cxx
  token_type_sysenv.cxx
  wrap_text_block.cxx
  )



# install generated header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_util_export.h
  SUBDIR   viame/algorithm_framework/util
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


# Republished under the prefix the imported sources use.
viame_lite_generated( vital_util_export.h algorithm_framework/util )
