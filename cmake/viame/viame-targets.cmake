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
#   viame_add_library( name [type] [sources...] [NO_VERSION] [NO_EXPORT_HEADER] )
#
# SHARED unless the caller names a type. The default without one is STATIC,
# which breaks transitive PRIVATE-dependency propagation for anything linking
# VIAME from outside.
#-
function( viame_add_library name )
  set( options NO_VERSION NO_EXPORT_HEADER NO_EXPORT )
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

  if( has_type )
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

  set( exports )
  if( NOT LIB_NO_EXPORT )
    set( exports EXPORT ${viame_export_name} )
    set_property( GLOBAL APPEND PROPERTY viame_export_targets ${name} )
  endif()

  install( TARGETS ${name} ${exports}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    COMPONENT runtime
    )

  if( NOT LIB_NO_EXPORT )
    set_property( GLOBAL APPEND PROPERTY viame_libraries ${name} )
  endif()
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
# Write the recorded targets to a file in the build tree.
#
# The namespace stays `kwiver::`. It is what an out-of-tree plugin writes --
# `examples/plugin_creation` links `kwiver::vital` -- and P5-T05 kept it
# deliberately when `viame-config.cmake` replaced kwiver's config package.
# Changing it here would be a rename of VIAME's public CMake surface, which
# is not what this task is.
#-
function( viame_export_targets file )
  get_property( targets GLOBAL PROPERTY viame_export_targets )
  export( TARGETS ${targets}
    NAMESPACE kwiver::
    ${ARGN}
    FILE "${file}"
    )
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
