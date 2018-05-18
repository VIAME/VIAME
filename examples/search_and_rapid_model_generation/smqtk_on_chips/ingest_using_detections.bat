:: Standard process of building IQR-required descriptors+models from scratch
:: and existing configs.

:: This assumes the use of the LSH nearest-neighbor index as it builds ITQ model.

call ..\..\..\setup_viame.bat

:: PARAMETERS ###################################################################

set IMAGE_LIST=input_list.txt
set IMAGE_TILES_DIR=tiles

:: Compute tiles using KWIVER pipeline
echo "Generating tiles"
mkdir -p "${IMAGE_TILES_DIR}"

pipeline_runner -p configs/chip_extractor_pipeline.pipe

:: Perform ingest on computed chips
call ingest_image_folder.bat %IMAGE_TILES_DIR%
