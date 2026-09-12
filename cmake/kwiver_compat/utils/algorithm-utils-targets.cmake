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
if( NOT TARGET all-plugins )
  add_custom_target( all-plugins )
endif()


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
# The named sources are compiled into the base library with their entry point
# renamed, and the base library is added to the generated static registry.
#
# Additional source files may be specified after the base library if the
# registration interface implementation is separate from the base library.
#
# Setting library_subdir or no_export before this function
# has no effect as they are manually specified within this function.
#-
function(algorithms_create_plugin    base_lib)
  # P8-T03: the registration file is compiled into the library it registers.
  #
  # This used to generate a small MODULE, `<base_lib>_plugin`, holding nothing
  # but the registration function, for the loader to find by scanning a
  # directory and `dlopen`. The generated registry calls that function
  # directly now, so there is nothing to find, and the module -- along with
  # the `_plugin` target that a few call sites used to add link libraries to
  # -- is gone.
  viame_register_statically( ${base_lib} ${ARGN} )
endfunction()
