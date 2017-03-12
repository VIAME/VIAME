###
# A common location for data used across multiple examples
##

if( VIAME_DOWNLOAD_MODELS )

  # Download small image set 1
  DownloadExtractAndInstall(
    https://data.kitware.com/api/v1/file/58c4b9458d777f0aef5d796e/download
    8a5a40018e3909d10b3fd50aad06e4f3
    ${VIAME_DOWNLOAD_DIR}/small_image_example_set1.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}
    examples/common_data
    small_example_image_set1 )

endif()
  
