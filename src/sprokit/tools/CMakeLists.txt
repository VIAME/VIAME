project(sprokit_tools)

sprokit_use_tools_fixes()

set(tools_srcs)

set(tools_headers
  tools-config.h)

sprokit_add_library(sprokit_tools
  ${tools_srcs}
  ${tools_headers})
set_target_properties(sprokit_tools
  PROPERTIES
    VERSION       ${sprokit_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_SPROKIT_TOOLS_LIB)
sprokit_require_tools_fixes(sprokit_tools)

sprokit_create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  tools)

sprokit_install_headers(sprokit/tools ${tools_headers})

sprokit_configure_pkgconfig(sprokit-tools)
