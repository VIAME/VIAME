set( EXAMPLE_INSTALL_FILES
     measurement_example.pipe
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
  https://data.kitware.com/api/v1/item/5ada33b18d777f0685794bf4/download
  c91d16b9b0ae4e59b3e0e54ac4369d4e
  ${VIAME_DOWNLOAD_DIR}/camtrawl_demodata.zip
  ${CMAKE_CURRENT_SOURCE_DIR} )

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/measurement_using_stereo )
install( DIRECTORY   camtrawl_demodata
         DESTINATION examples/measurement_using_stereo )
