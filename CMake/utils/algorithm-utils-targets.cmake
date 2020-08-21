#
# ARROWS Target creation and installation support
#
# Variables that affect behavior of functions:
#
#   no_export
#       if set, target will not be exported.
#
#   no_install
#       If set, target will not be installed.
#
#   no_version
#       If set, the target will not have version information added to it.
#
#   component
#     If set, the target will not be installed under this component (the
#     default is 'runtime').
#
#   library_subdir
#     If set, library targets will be placed into the directory with this
#     as a suffix. This is necessary due to the way some systems use
#     CMAKE_BUILD_TYPE as a directory in the output path.
#
include(CMakeParseArguments)

# Global collection variables
define_property(GLOBAL PROPERTY arrows_plugin_libraries
  BRIEF_DOCS "Generated plugin libraries"
  FULL_DOCS "List of generated shared plugin module libraries"
  )

define_property(GLOBAL PROPERTY arrows_bundle_paths
  BRIEF_DOCS "Paths needed by fixup_bundle"
  FULL_DOCS "Paths needed to resolve needed libraries used by plugins when fixing the bundle"
  )

# Top-level target for plugin targets
add_custom_target( all-plugins )


#+
# Generate and add a plug-in library based on another library
#
#   algorithms_create_plugin(base_lib [args ...])
#
# The given base library must link against the core arrows library and provide
# an implementation of the algorithm plugin interface class. If this has not
# been done an error will occur at link time stating that the required class
# symbol can not be found.
#
# This generates a small MODULE library that exposes the required C interface
# function to be picked up by the algorithm plugin manager. This library is set
# to install into the .../arrows subdirectory and adds a _plugin suffix to the
# base library name.
#
# Additional source files may be specified after the base library if the
# registration interface implementation is separate from the base library.
#
# Setting library_subdir or no_export before this function
# has no effect as they are manually specified within this function.
#-
function(algorithms_create_plugin    base_lib)
  message( STATUS "Building plugin \"${base_lib}\"" )

  # Make a plugin from the supplied files. The name here is largely
  # irrelevant since they are discovered at run time.
  set( plugin_name   "${base_lib}_plugin" )

  # create module library given generated source, linked to given library
  set(library_subdir /${kwiver_plugin_algorithm_subdir})
  set(no_version ON)

  kwiver_add_plugin( ${plugin_name}
    SOURCES  ${ARGN}
    # Not adding link to known base library because if the base_lib isn't
    # linking against it, its either doing something really complex or doing
    # something wrong (most likely the wrong).
    PRIVATE  ${base_lib}
    )

  set_target_properties( ${plugin_name}
    PROPERTIES
    OUTPUT_NAME   ${base_lib}_plugin
    INSTALL_RPATH "\$ORIGIN/../../lib:\$ORIGIN/"
    )

  add_dependencies( all-plugins ${plugin_name} )

  # For each library linked to the base library, add the path to the library
  # to a list of paths to search later during fixup_bundle.
  # Recursively add paths for dependencies of these libraries which are targets.
  get_target_property(deps ${base_lib} LINK_LIBRARIES)
  while(deps)
    unset(rdeps)
    foreach( dep ${deps} )
      if(TARGET "${dep}")
        list(APPEND PLUGIN_BUNDLE_PATHS $<TARGET_FILE_DIR:${dep}>)
        get_target_property(target_type ${dep} TYPE)
        if (NOT ${target_type} STREQUAL "INTERFACE_LIBRARY")
          get_target_property(recursive_deps ${dep} LINK_LIBRARIES)
          if(recursive_deps)
            list(APPEND rdeps ${recursive_deps})
          endif()
        endif()
      elseif(EXISTS "${dep}")
        get_filename_component(dep_dir "${dep}" DIRECTORY)
        list(APPEND PLUGIN_BUNDLE_PATHS ${dep_dir})
      endif()
    endforeach()
    set(deps ${rdeps})
  endwhile()

  # Add to global collection variables
  set_property(GLOBAL APPEND
    PROPERTY arrows_plugin_libraries    ${plugin_name}
    )
  set_property(GLOBAL APPEND
    PROPERTY arrows_bundle_paths ${PLUGIN_BUNDLE_PATHS}
    )

endfunction()
