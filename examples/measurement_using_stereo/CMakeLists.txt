set( EXAMPLE_INSTALL_FILES
     README.rst )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_measurer.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_measurer.sh )
endif()

DownloadAndExtract(
  https://data.kitware.com/api/v1/item/5ada492b8d777f0685794c86/download
  fef401788fbba63d530ae6fda05b129b
  ${VIAME_DOWNLOAD_DIR}/camtrawl_set1.tar.gz
  ${CMAKE_CURRENT_SOURCE_DIR} )

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/measurement_using_stereo )
install( DIRECTORY   camtrawl_data
         DESTINATION examples/measurement_using_stereo )
