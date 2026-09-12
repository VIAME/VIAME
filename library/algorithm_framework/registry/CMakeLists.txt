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
  "${generated}"
  )

viame_lite_generated( viame_registry_export.h algorithm_framework/registry )

# PRIVATE: the registry names one symbol in each registered library and
# nothing else, and nothing that links the registry needs their headers. It
# is still enough to put them in the process -- a referenced symbol is what
# `--as-needed` keeps -- and it keeps libraries that are not exported out of
# the registry's install interface.
target_link_libraries( viame_registry
  PRIVATE              kwiver::vital_vpm
                       kwiver::vital_logger
                       ${CMAKE_DL_LIBS}
                       ${viame_registered_libraries}
  )

set_target_properties( viame_registry PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

###
# What executables link.
#
# Nothing calls into `viame_registry` by name -- it hands itself to the plugin
# manager as it loads -- so the toolchain's `--as-needed` would drop it from
# every link line as unused, and with it every factory in VIAME. The flags
# around it say to keep it, and are scoped to it so the rest of the link line
# is unaffected.
##
add_library( viame_registry_linked INTERFACE )

if( CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT APPLE )
  target_link_libraries( viame_registry_linked INTERFACE
    "-Wl,--no-as-needed" viame_registry "-Wl,--as-needed" )
else()
  # The Apple linker keeps what it is given, and MSVC has no equivalent of
  # `--as-needed` for shared libraries.
  target_link_libraries( viame_registry_linked INTERFACE viame_registry )
endif()

add_library( viame::registry ALIAS viame_registry_linked )

install( TARGETS viame_registry_linked EXPORT ${kwiver_export_name} )
set_property( GLOBAL APPEND PROPERTY viame_export_targets viame_registry_linked )

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
