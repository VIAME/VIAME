# miniz, vendored, in place of zlib
#
# Two files of the amalgamation and its licence. Not the examples, the bazel
# build, the changelog or the readme.
#
# VIAME asks zlib for three things and no more: raw deflate, raw inflate and
# crc32. `library/utilities/utilities_file.cxx` writes NPZ -- which is a ZIP of
# `.npy` members -- and `camera_rig_io.cxx` reads it back. Both build and
# parse the ZIP container themselves, so none of an archive library is
# wanted, only the codec inside it.
#
# So miniz is compiled with everything else switched off:
#
#   MINIZ_NO_ARCHIVE_APIS   the zip reader and writer, which VIAME does itself
#   MINIZ_NO_STDIO          no FILE* paths; the callers hold buffers
#   MINIZ_NO_TIME           only the archive APIs date anything
#
# The zlib-compatible names stay on, so `utilities_file.cxx` and
# `camera_rig_io.cxx` call `deflateInit2`, `inflate` and `crc32` exactly as
# they did against zlib. Only their `#include` changed.

set( CMAKE_POSITION_INDEPENDENT_CODE TRUE )

add_library( viame_miniz STATIC miniz.c miniz.h )

target_include_directories( viame_miniz PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

target_compile_definitions( viame_miniz PUBLIC
  MINIZ_NO_ARCHIVE_APIS
  MINIZ_NO_STDIO
  MINIZ_NO_TIME
  )

set_target_properties( viame_miniz PROPERTIES
  OUTPUT_NAME miniz
  )

add_library( viame::miniz ALIAS viame_miniz )

# Not installed: its code is linked into libviame, which is. It was installed
# only to join the export set, and nothing is exported any more.
