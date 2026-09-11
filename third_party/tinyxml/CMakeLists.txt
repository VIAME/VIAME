# TinyXML 1, vendored
#
# Four files: the header and the three translation units. `tinystr.{h,cpp}`
# are not carried, because they are only compiled when TinyXML is built
# **without** STL strings and VIAME has always built it with them -- under
# `TIXML_USE_STL` the header does not so much as include tinystr.h. The
# distribution's documentation, test program, sample XML and Visual Studio
# project files are not carried either.
#
# One caller: `plugins/core/read_detected_object_set_cvat.cxx`. The CVAT
# reader is the only thing in VIAME that parses XML, and
# `lite-removals.md` leaves it open whether P8 replaces TinyXML with a small
# subset reader of our own; vendoring does not decide that, it only stops
# fletch having to build a library for one file.
#
# TinyXML is zlib-licensed; the notice is the comment at the top of
# `tinyxml.h` and travels with it.

set( CMAKE_POSITION_INDEPENDENT_CODE TRUE )

add_library( viame_tinyxml STATIC
  tinyxml.cpp
  tinyxmlerror.cpp
  tinyxmlparser.cpp
  tinyxml.h
  )

target_include_directories( viame_tinyxml PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

# What VIAME has always built it with, and what lets `tinystr` go
target_compile_definitions( viame_tinyxml PUBLIC TIXML_USE_STL )

set_target_properties( viame_tinyxml PROPERTIES
  OUTPUT_NAME tinyxml
  )

add_library( viame::tinyxml ALIAS viame_tinyxml )
