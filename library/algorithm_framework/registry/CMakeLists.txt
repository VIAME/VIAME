###
# The static plugin registry
#
# Added from the top-level CMakeLists rather than from the algorithm
# framework, because it has to come last: every library that registers has
# already called `viame_register_statically`, and the generated file below
# names them all.
#
# See `cmake/viame_static_registration.cmake` for the mechanism and
# `static_registry.h` for why there is one.
##

set( generated "${CMAKE_CURRENT_BINARY_DIR}/static_registry.cxx" )
viame_write_static_registry( "${generated}" )

get_property( viame_registered_libraries
  GLOBAL PROPERTY viame_static_registrations )
list( REMOVE_DUPLICATES viame_registered_libraries )

viame_install_headers(
  SUBDIR     viame/algorithm_framework/registry
  external_plugins.h
  static_registry.h
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_registry_export.h
  NOPATH   SUBDIR     viame/algorithm_framework/registry
  )

viame_add_library( viame_registry
  external_plugins.h
  external_plugins.cxx
  static_registry.h
  )

viame_lite_generated( viame_registry_export.h algorithm_framework/registry )

# PRIVATE: the registry names one symbol in each registered library and
# nothing else, and nothing that links the registry needs their headers.
# What puts their code in the process is `viame_library` below, which links
# every folded library.
target_link_libraries( viame_registry
  PRIVATE              kwiver::vital_vpm
                       kwiver::vital_logger
                       ${CMAKE_DL_LIBS}
                       ${viame_registered_libraries}
  )

###
# libviame
#
# Every folded library, linked into one shared library -- `viame_add_library`
# made each an OBJECT library, and this is where their code ends up. The
# generated registry is its own source, since it names a function in every
# one of them.
#
# PRIVATE, so their objects are linked here and only here; CMake copies an
# object library's code into the targets that link it directly and no
# further. Their usage requirements still reach whatever links this in the
# build tree -- include paths, and the libraries each one links -- through
# the INTERFACE entries, which are build-only because nothing but libviame
# itself is installed.
##
get_property( viame_folded_libraries GLOBAL PROPERTY viame_folded_libraries )

viame_add_library( viame_library
  "${generated}"
  NO_FOLD NO_EXPORT_HEADER
  )

set_target_properties( viame_library PROPERTIES
  OUTPUT_NAME          viame
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

# The generated registry is compiled here rather than in `viame_registry`,
# whose export header decides what it exports.
target_compile_definitions( viame_library PRIVATE viame_registry_EXPORTS )

target_link_libraries( viame_library
  PRIVATE              ${viame_folded_libraries}
                       ${CMAKE_DL_LIBS}
  )

foreach( _viame_folded IN LISTS viame_folded_libraries )
  target_link_libraries( viame_library
    INTERFACE          $<BUILD_INTERFACE:${_viame_folded}> )
endforeach()

if( APPLE AND KWIVER_ENABLE_PYTHON )
  set_target_properties( viame_library PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup" )
endif()

list( LENGTH viame_folded_libraries _viame_folded_count )
message( STATUS "libviame: ${_viame_folded_count} libraries folded" )

###
# What executables link.
#
# Nothing calls into the registry by name -- it hands itself to the plugin
# manager as libviame loads -- so the toolchain's `--as-needed` would drop
# libviame from a link line that references nothing else in it, and with it
# every factory in VIAME. The flags around it say to keep it, and are scoped
# to it so the rest of the link line is unaffected.
##
add_library( viame_registry_linked INTERFACE )

if( CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT APPLE )
  target_link_libraries( viame_registry_linked INTERFACE
    "-Wl,--no-as-needed" viame_library "-Wl,--as-needed" )
else()
  # The Apple linker keeps what it is given, and MSVC has no equivalent of
  # `--as-needed` for shared libraries.
  target_link_libraries( viame_registry_linked INTERFACE viame_library )
endif()

add_library( viame::registry ALIAS viame_registry_linked )

# Not installed or exported: an out-of-tree plugin gets `kwiver::viame` and
# the names it has always used from the config package, which
# `cmake/viame-config-targets-install.cmake.in` writes.

###
# The python extensions that load plugins
#
# `kwiver.vital.modules` and `kwiver.vital.plugin_management` both call
# `load_all_plugins`, so both need the registry -- but they are built in
# `python/`, which has to come before this directory because the python
# plugin module registers into it. So the link is made from here, where both
# ends exist, rather than from there, where one of them does not.
##
foreach( _viame_registry_consumer
    python-vital.modules-modules
    python-vital.plugin_management-_plugin_management )

  if( TARGET ${_viame_registry_consumer} )
    target_link_libraries( ${_viame_registry_consumer}
      PRIVATE viame_registry_linked )
  endif()
endforeach()
