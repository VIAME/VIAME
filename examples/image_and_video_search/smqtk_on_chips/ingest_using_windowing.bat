:: Standard process of building IQR-required descriptors+models from scratch
:: and existing configs.

:: This assumes the use of the LSH nearest-neighbor index as it builds ITQ model.

call ..\..\..\setup_viame.bat

:: PARAMETERS ###################################################################

set IMAGE_LIST=input_list.txt
set IMAGE_TILES_DIR=tiles

set SMQTK_GEN_IMG_TILES=configs\generate_image_transform.tiles.json

:: Compute tiles via chipping up the input images in a grid-like pattern
for /F "tokens=*" %%L in (%IMAGE_DIR_FILELIST%) do generate_image_transform -c %%L -i %%L -o %IMAGE_TILES_DIR%

:: Perform ingest on computed chips
call ingest_image_folder.bat %IMAGE_TILES_DIR%
