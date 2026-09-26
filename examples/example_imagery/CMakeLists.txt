###
# A common location for data used across multiple examples
##

if( VIAME_DOWNLOAD_MODELS )

  # Download small image set 1
  DownloadExtractAndInstall(
    https://viame.kitware.com/api/v1/item/6ab572e92d17596fb9c5d3e6/download
    8a5a40018e3909d10b3fd50aad06e4f3
    ${VIAME_DOWNLOAD_DIR}/small_image_example_set1.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}
    examples/example_imagery
    small_example_image_set1 )

  # Download habcam image set 1
  DownloadExtractAndInstall(
    https://viame.kitware.com/api/v1/item/6ab572de2d17596fb9c5d3ce/download
    ac40557e8fea502fb988bd3d934c51f1
    ${VIAME_DOWNLOAD_DIR}/habcam_example_image_set1.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}
    examples/example_imagery
    habcam_example_image_set1 )

  # Download mouss image set 1
  DownloadExtractAndInstall(
    https://viame.kitware.com/api/v1/item/6ab572e52d17596fb9c5d3de/download
    ea8f4f37d14c4a3e3efead045effe23b
    ${VIAME_DOWNLOAD_DIR}/mouss_example_image_set1.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}
    examples/example_imagery
    mouss_example_image_set1 )

  # Download camtrawl image set 1
  DownloadExtractAndInstall(
    https://viame.kitware.com/api/v1/item/6ab572d52d17596fb9c5d3ca/download
    d54ccf4463ec271f4e63922c10a235bd
    ${VIAME_DOWNLOAD_DIR}/camtrawl_example_image_set1.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}
    examples/example_imagery
    camtrawl_example_image_set1 )

  # Download auv image set 1
  DownloadExtractAndInstall(
    https://viame.kitware.com/api/v1/item/6ab572e52d17596fb9c5d3e2/download
    e5ddfa558613232cdaa354bad42d65e6
    ${VIAME_DOWNLOAD_DIR}/raw_auv_image_set1.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}
    examples/example_imagery
    raw_auv_image_set1 )

  # Download arctic seal aerial imagery
  if( VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL OR VIAME_ENABLE_OPENCV )
    DownloadExtractAndInstall(
      https://viame.kitware.com/api/v1/item/6ab572d42d17596fb9c5d3c6/download
      0491109936bb39457e1540d6a85bc659
      ${VIAME_DOWNLOAD_DIR}/arctic_seal_example_set1.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}
      examples/example_imagery
      arctic_seal_example_set1 )
  endif()

endif()

