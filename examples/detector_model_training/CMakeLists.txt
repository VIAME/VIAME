###
# Model training examples, old API
##

set( example_files
  format_data_for_training.py )

install( FILES       ${example_files}
         DESTINATION examples/detector_model_training )

if( VIAME_DOWNLOAD_MODELS )
  DownloadExtractAndInstall(
    https://data.kitware.com/api/v1/file/58d325ca8d777f0aef5d88ed/download
    b5497368fb79c74f8172213b14a08f7d
    ${VIAME_DOWNLOAD_DIR}/training_data_set1.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}
    examples/detector_model_training
    training_data )
endif()
