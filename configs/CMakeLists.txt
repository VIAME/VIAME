###
# Root configuration file CMakeLists
##

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
