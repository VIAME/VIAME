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

  # Download habcam image set 1
  DownloadExtractAndInstall(
    https://data.kitware.com/api/v1/item/595a80058d777f16d01e993d/download
    ac40557e8fea502fb988bd3d934c51f1
    ${VIAME_DOWNLOAD_DIR}/habcam_example_image_set1.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}
    examples/common_data
    habcam_example_image_set1 )

  # Download mouss image set 1
  DownloadExtractAndInstall(
    https://data.kitware.com/api/v1/item/59ae1b2d8d777f7d33e9cf30/download
    ea8f4f37d14c4a3e3efead045effe23b
    ${VIAME_DOWNLOAD_DIR}/mouss_example_image_set1.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}
    examples/common_data
    mouss_example_image_set1 )

  # Download default caffe classification models
  if( VIAME_ENABLE_CAFFE )
    DownloadExtractAndInstall(
      https://data.kitware.com/api/v1/item/5a83a7d88d777f0685782a56/download
      2cdbe7ca51b49b12e59911aadec20e98
      ${VIAME_DOWNLOAD_DIR}/default_caffe_models_v0.1.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}
      examples/common_data
      models )
  endif()

endif()

