
# Validate that all required dependencies are enabled for selected addons
ValidateAddonDependencies()

# Process downloads and install models/transforms for all enabled addons
ProcessAllAddonModels( "${CMAKE_CURRENT_SOURCE_DIR}" )

# Install pipeline files from addon subdirectories if they exist
# For each enabled VIAME_DOWNLOAD_MODELS-NAME, check if configs/add-ons/name exists
# (where name is lowercase with dashes preserved) and install its contents
foreach( _addon_name IN LISTS VIAME_ADDON_NAMES )
  if( VIAME_DOWNLOAD_MODELS-${_addon_name} )
    string( TOLOWER "${_addon_name}" _addon_dir_name )
    set( _addon_dir "${CMAKE_CURRENT_SOURCE_DIR}/${_addon_dir_name}" )
    if( IS_DIRECTORY "${_addon_dir}" )
      install( DIRECTORY "${_addon_dir}/"
               DESTINATION configs/pipelines )
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
