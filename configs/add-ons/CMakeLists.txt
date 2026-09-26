
# Validate that all required dependencies are enabled for selected addons
ValidateAddonDependencies()

# Process downloads and install models/transforms for all enabled addons
ProcessAllAddonModels( "${CMAKE_CURRENT_SOURCE_DIR}" )

# Install each enabled add-on's pipeline files into configs/pipelines, with
# later add-ons overriding earlier ones and the core defaults by install order.
# We can't use install(DIRECTORY)/file(COPY): they skip a copy when the
# destination isn't older than the source, and these files share a git-checkout
# timestamp, so overrides get dropped. Force each copy (remove then copy) so
# install order alone decides the winner. The walk is per file rather than per
# top level entry: removing a directory wholesale would take the destination's
# other contents with it, which for a models directory means every weight
# already downloaded there. Directories therefore merge, and same named files
# are still overwritten.
foreach( _addon_name IN LISTS VIAME_ADDON_NAMES )
  if( VIAME_DOWNLOAD_MODELS-${_addon_name} )
    string( TOLOWER "${_addon_name}" _addon_dir_name )
    set( _addon_dir "${CMAKE_CURRENT_SOURCE_DIR}/${_addon_dir_name}" )
    if( IS_DIRECTORY "${_addon_dir}" )
      install( CODE "
        file( GLOB_RECURSE _addon_files RELATIVE \"${_addon_dir}\" \"${_addon_dir}/*\" )
        foreach( _addon_rel IN LISTS _addon_files )
          get_filename_component( _addon_rel_dir \"\${_addon_rel}\" DIRECTORY )
          file( REMOVE
                \"\${CMAKE_INSTALL_PREFIX}/configs/pipelines/\${_addon_rel}\" )
          file( COPY \"${_addon_dir}/\${_addon_rel}\"
                DESTINATION
                \"\${CMAKE_INSTALL_PREFIX}/configs/pipelines/\${_addon_rel_dir}\" )
        endforeach()
      " )
    endif()
  endif()
endforeach()

# Emit pipe-file comparison warnings at the end of the add-ons install phase
get_property( _pipe_warnings GLOBAL PROPERTY VIAME_PIPE_WARNINGS )
if( _pipe_warnings )
  set( _warning_code "" )
  foreach( _warn IN LISTS _pipe_warnings )
    string( APPEND _warning_code "message( WARNING \"${_warn}\" )\n" )
  endforeach()
  install( CODE "${_warning_code}" )
endif()
