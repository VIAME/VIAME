###
# Root configuration file CMakeLists
##

# Nothing in the install ever removes a file, so a pipeline that is renamed or
# deleted in the source tree stays in configs/pipelines forever. That is not
# merely untidy: the pipeline tests enumerate the install directory, so stale
# files become tests for pipelines that no longer exist, and they keep passing
# against models left behind by earlier add-on packs. Clear the .pipe files
# first and let the install rules below put back exactly what is current.
#
# Declared before the subdirectories so it runs before anything installs into
# that directory. Only *.pipe is touched -- models/ and transformations/ hold
# downloaded weights that must survive, and templates/ is reinstalled wholesale.
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
if( VIAME_DOWNLOAD_MODELS AND VIAME_ENABLE_VIVIA )
  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/5afe6c018d777f15ebe1d6f2/download
    7e8800cb11a1a491140e7f58e9053030
    ${VIAME_DOWNLOAD_DIR}/models-blue_marble_context-v1.0.0.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}/gui-params )
endif()
