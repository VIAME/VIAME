# VIAME's library, plugin and executable helpers
#
# These were kwiver's `kwiver_add_*` until P8-T08. Two things went with the
# rename.
#
# **The branches VIAME's build never takes are gone.** `SKBUILD` is never
# set here -- VIAME is not built through scikit-build -- so the two library
# destinations it chose between are one. `LIB_SUFFIX` is empty and
# `library_dir` is always `lib`, so the path arithmetic that computed a
# relative path back to the prefix from an arbitrary library directory --
# three helper functions, sixty lines -- is `..`. `library_subdir` was
# emptied by P8-T03 when plugins stopped being modules in a scanned
# directory, so every library lands in `lib/` and the RPATH is the same for
# all of them. What is kept is the multi-config block: VIAME's Windows CI
# uses Visual Studio, this machine cannot run it, and a generator expression
# that is wrong there would not show up here.
#
# **The knobs no caller turns are gone.** `no_export`, `no_install`,
# `no_version`, `no_export_header` and `component` were read out of the
# caller's directory scope -- set a variable, call the function, hope the
# reader notices. A grep of every `CMakeLists.txt` in the tree finds none of
# them set except inside `viame_add_plugin`, which sets `no_version` for its
# own call. They are keyword arguments now, so the call site says what it
# means.

include_guard( GLOBAL )

include( GenerateExportHeader )

#+
# Add a library.
#
#   viame_add_library( name [type] [sources...] [NO_VERSION] [NO_EXPORT_HEADER]
#                      [NO_FOLD] )
#
# SHARED unless the caller names a type. The default without one is STATIC,
# which breaks transitive PRIVATE-dependency propagation for anything linking
# VIAME from outside.
#
# **Folding.** With `VIAME_FOLD_LIBRARIES` on, a library that would be SHARED
# is an OBJECT library instead, and `library/algorithm_framework/registry`
# links every one of them into the single `libviame`. VIAME's own configure
# turns it on; an out-of-tree plugin including this file from the install
# does not, and gets the shared library it asked for. `NO_FOLD` is for
# `libviame` itself. A folded library is not installed -- its code is -- and
# it registers nothing on its own, so a target that is not folded links it
# through `viame_target_link_libraries`.
#
# Nothing is exported. `viame-config-targets.cmake` is written from
# `cmake/viame-config-targets-install.cmake.in`, because an export set would
# have had to describe the folded OBJECT libraries.
#-
function( viame_add_library name )
  set( options NO_VERSION NO_EXPORT_HEADER NO_FOLD )
  cmake_parse_arguments( LIB "${options}" "" "" ${ARGN} )

  string( TOUPPER "${name}" upper_name )

  set( has_type FALSE )
  foreach( arg IN LISTS LIB_UNPARSED_ARGUMENTS )
    if( arg STREQUAL "STATIC" OR arg STREQUAL "SHARED" OR
        arg STREQUAL "MODULE" OR arg STREQUAL "OBJECT" OR
        arg STREQUAL "INTERFACE" )
      set( has_type TRUE )
      break()
    endif()
  endforeach()

  set( fold FALSE )
  if( VIAME_FOLD_LIBRARIES AND NOT LIB_NO_FOLD )
    if( NOT has_type OR "SHARED" IN_LIST LIB_UNPARSED_ARGUMENTS )
      set( fold TRUE )
    endif()
  endif()

  if( fold )
    set( sources ${LIB_UNPARSED_ARGUMENTS} )
    list( REMOVE_ITEM sources SHARED )
    add_library( ${name} OBJECT ${sources} )
  elseif( has_type )
    add_library( ${name} ${LIB_UNPARSED_ARGUMENTS} )
  else()
    add_library( ${name} SHARED ${LIB_UNPARSED_ARGUMENTS} )
  endif()

  if( APPLE )
    set( version_props
      MACOSX_RPATH      TRUE
      INSTALL_NAME_DIR  "@executable_path/../lib"
      )
  elseif( LIB_NO_VERSION )
    set( version_props )
  else()
    set( version_props
      VERSION    ${${CMAKE_PROJECT_NAME}_VERSION}
      SOVERSION  ${${CMAKE_PROJECT_NAME}_VERSION}
      )
  endif()

  set_target_properties( ${name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY      "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY      "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY      "${CMAKE_BINARY_DIR}/bin"
    INSTALL_RPATH                 "\$ORIGIN/../lib:\$ORIGIN/"
    INTERFACE_INCLUDE_DIRECTORIES
      "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR};${CMAKE_BINARY_DIR}>$<INSTALL_INTERFACE:include>"
    ${version_props}
    )

  # Visual Studio and the other multi-config generators put each
  # configuration in its own directory. Not exercised on this machine; the
  # Windows CI is the only thing that reads these.
  foreach( config IN LISTS CMAKE_CONFIGURATION_TYPES )
    string( TOUPPER "${config}" upper_config )
    set_target_properties( ${name} PROPERTIES
      "ARCHIVE_OUTPUT_DIRECTORY_${upper_config}" "${CMAKE_BINARY_DIR}/lib/${config}"
      "LIBRARY_OUTPUT_DIRECTORY_${upper_config}" "${CMAKE_BINARY_DIR}/lib/${config}"
      "RUNTIME_OUTPUT_DIRECTORY_${upper_config}" "${CMAKE_BINARY_DIR}/bin/${config}"
      )
  endforeach()

  if( NOT LIB_NO_EXPORT_HEADER )
    generate_export_header( ${name}
      STATIC_DEFINE ${upper_name}_BUILD_AS_STATIC
      )
  endif()

  get_target_property( target_type ${name} TYPE )
  if( target_type STREQUAL "STATIC_LIBRARY" )
    set_target_properties( ${name} PROPERTIES POSITION_INDEPENDENT_CODE TRUE )
  endif()

  if( fold )
    set_target_properties( ${name} PROPERTIES POSITION_INDEPENDENT_CODE TRUE )

    # What CMake defines for a shared library's own sources and the generated
    # export header keys on. An OBJECT library gets no such definition, and
    # without it every symbol would be an import on Windows.
    target_compile_definitions( ${name} PRIVATE ${name}_EXPORTS )

    set_property( GLOBAL APPEND PROPERTY viame_folded_libraries ${name} )
    return()
  endif()

  install( TARGETS ${name}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    COMPONENT runtime
    )
endfunction()

#+
# Add an executable.
#
#   viame_add_executable( name [sources...] )
#-
function( viame_add_executable name )
  add_executable( ${name} ${ARGN} )

  set_target_properties( ${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    INSTALL_RPATH            "\$ORIGIN/../lib:\$ORIGIN/"
    )

  set_property( GLOBAL APPEND PROPERTY viame_executables "${name}" )
  set_property( GLOBAL APPEND
    PROPERTY viame_executables_paths "${CMAKE_CURRENT_BINARY_DIR}" )

  install( TARGETS ${name}
    DESTINATION bin
    COMPONENT   runtime
    )
endfunction()

#+
# Install headers under `include/`.
#
#   viame_install_headers( [headers...] [SUBDIR dir] [NOPATH] )
#
# NOPATH flattens: a generated header in the build tree installs beside the
# hand-written ones rather than under the directory it was generated in.
#-
function( viame_install_headers )
  set( options NOPATH )
  set( oneValueArgs SUBDIR )
  cmake_parse_arguments( HEADER "${options}" "${oneValueArgs}" "" ${ARGN} )

  foreach( header IN LISTS HEADER_UNPARSED_ARGUMENTS )
    if( HEADER_NOPATH )
      set( subdir )
    else()
      get_filename_component( subdir "${header}" DIRECTORY )
      set( subdir "/${subdir}" )
    endif()

    install( FILES "${header}"
      DESTINATION "include/${HEADER_SUBDIR}${subdir}"
      )
  endforeach()

  source_group( "Header Files\\Public" FILES ${HEADER_UNPARSED_ARGUMENTS} )
endfunction()

#+
# Group headers for the IDE without installing them.
#-
function( viame_private_header_group )
  source_group( "Header Files\\\\Private" FILES ${ARGN} )
endfunction()

#+
# Link a target that is not folded against VIAME's libraries.
#
#   viame_target_link_libraries( target [PUBLIC|PRIVATE|INTERFACE] items... )
#
# For executables, tests, python extensions and loadable modules. Outside a
# folded build this is `target_link_libraries`.
#
# Inside one, a directly linked OBJECT library has its code copied into the
# target, beside the copy in `libviame` -- two plugin managers, two loggers,
# every factory registered twice. So the call is recorded on the target and
# made by `viame_apply_folded_links` at the end of the configure, once every
# library exists and it can be told which names were folded. Deferred rather
# than resolved here because a test in `library/<dir>/tests` is defined
# before the libraries added after its directory, the registry among them.
#-
function( viame_target_link_libraries target )
  if( NOT VIAME_FOLD_LIBRARIES )
    target_link_libraries( ${target} ${ARGN} )
    return()
  endif()

  set_property( TARGET ${target} APPEND PROPERTY VIAME_DEFERRED_LINKS ${ARGN} )
  set_property( GLOBAL APPEND PROPERTY viame_deferred_link_targets ${target} )
endfunction()

#+
# Make the links `viame_target_link_libraries` recorded.
#
# A name that is, or is an alias of, a folded library becomes
# `viame_registry_linked` -- `libviame`, kept on the link line even where
# nothing references it by name, because the registry in it is what the
# target was linking a plugin library for. Everything else is passed through.
#-
function( viame_apply_folded_links )
  get_property( targets GLOBAL PROPERTY viame_deferred_link_targets )
  get_property( folded GLOBAL PROPERTY viame_folded_libraries )

  if( NOT targets )
    return()
  endif()

  list( REMOVE_DUPLICATES targets )

  foreach( target IN LISTS targets )
    get_target_property( items ${target} VIAME_DEFERRED_LINKS )

    set( keyword PRIVATE )
    set( resolved )

    foreach( item IN LISTS items )
      if( item MATCHES "^(LINK_)?(PUBLIC|PRIVATE|INTERFACE)$" )
        if( resolved )
          target_link_libraries( ${target} ${keyword} ${resolved} )
          set( resolved )
        endif()
        set( keyword ${CMAKE_MATCH_2} )
        continue()
      endif()

      set( real "${item}" )
      if( TARGET "${item}" )
        get_target_property( aliased "${item}" ALIASED_TARGET )
        if( aliased )
          set( real "${aliased}" )
        endif()
      endif()

      if( real IN_LIST folded )
        if( NOT "viame_registry_linked" IN_LIST resolved )
          list( APPEND resolved viame_registry_linked )
        endif()
      else()
        list( APPEND resolved "${item}" )
      endif()
    endforeach()

    if( resolved )
      target_link_libraries( ${target} ${keyword} ${resolved} )
    endif()
  endforeach()
endfunction()

#+
# Add a plugin: an ordinary shared library that the generated registry links.
#
#   viame_add_plugin( name SOURCES ... [PUBLIC ...] [PRIVATE ...] [KIND k] )
#
# Until P8-T03 a plugin was a MODULE dropped in a subdirectory for the loader
# to find by scanning and `dlopen`, and `SUBDIR` said which one. Nothing
# scans now -- the registry calls the registration function directly, which
# is why the library has to be a real one, since a MODULE cannot be linked.
#
# What the directory was actually telling the loader was the plugin's *kind*,
# and `load_all_plugins` still takes a mask of those. P8-T03 read the kind
# back out of `SUBDIR` by matching the tail of the path; P8-T08 asks for it,
# because a path that has to be pattern-matched to recover a meaning it no
# longer has is a path that should have been the meaning.
#-
function( viame_add_plugin name )
  set( oneValueArgs KIND )
  set( multiValueArgs SOURCES PUBLIC PRIVATE )
  cmake_parse_arguments( PLUGIN "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  if( NOT PLUGIN_KIND )
    set( PLUGIN_KIND OTHERS )
  endif()

  # Exported, unlike the modules these were: the registry links them, and a
  # PRIVATE dependency of an exported shared library still has to be in the
  # export set for `install( EXPORT )` to be able to describe it.
  #
  # NO_VERSION because nothing loads them by soname.
  viame_add_library( ${name} SHARED ${PLUGIN_SOURCES} NO_VERSION )

  target_link_libraries( ${name}
    PUBLIC   ${PLUGIN_PUBLIC}
    PRIVATE  ${PLUGIN_PRIVATE}
    )

  viame_mark_static_registration( ${name} ${PLUGIN_KIND} ${PLUGIN_SOURCES} )

  set_property( GLOBAL APPEND PROPERTY viame_plugin_libraries ${name} )
endfunction()
