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
  https://data.kitware.com/api/v1/item/5a8607858d777f068578345e/download
  0dc91b4cac3e2b32051c69500bb1d1d3
  ${VIAME_DOWNLOAD_DIR}/camtrawl_demodata.zip
  ${CMAKE_CURRENT_SOURCE_DIR} )

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/measurement_using_stereo )
install( DIRECTORY   camtrawl_demodata
         DESTINATION examples/measurement_using_stereo )
