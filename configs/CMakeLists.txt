###
# Root configuration file CMakeLists
##

# Installs never remove files, so clear stale pipelines before reinstalling.
# Only *.pipe and *.conf; models/ and transformations/ hold downloaded weights.
install( CODE "
  file( GLOB_RECURSE _viame_stale_pipes
        \"\${CMAKE_INSTALL_PREFIX}/configs/pipelines/*.pipe\"
        \"\${CMAKE_INSTALL_PREFIX}/configs/pipelines/*.conf\" )
  list( LENGTH _viame_stale_pipes _viame_stale_count )
  if( _viame_stale_count GREATER 0 )
    message( STATUS
      \"Clearing \${_viame_stale_count} previously installed pipeline file(s)\" )
    file( REMOVE \${_viame_stale_pipes} )
  endif()
" )

# Core, default configuration files
add_subdirectory( pipelines )

# Optional, project-specific add-ons
if( VIAME_DOWNLOAD_MODELS )
  add_subdirectory( add-ons )
endif()

# Default GUI config parameter files
install( DIRECTORY gui-params
         DESTINATION configs )

# Project folders (groupings of config files for multiple tasks)
if( WIN32 )
  install( DIRECTORY prj-windows/
           DESTINATION configs/prj-template )
else()
  install( DIRECTORY prj-linux/
           DESTINATION configs/prj-template )
endif()

# GUI background context view
# The VIEW background context view was downloaded here, with VIVIA.
