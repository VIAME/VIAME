###
# Root configuration file CMakeLists
##

# Installs never remove files: clear what the previous install put in
# configs/pipelines before reinstalling (see the script for what qualifies).
install( CODE "set( VIAME_BUILD_MANIFEST \"${CMAKE_BINARY_DIR}/install_manifest.txt\" )" )
install( SCRIPT "${VIAME_CMAKE_DIR}/prune_installed_pipelines.cmake" )

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
