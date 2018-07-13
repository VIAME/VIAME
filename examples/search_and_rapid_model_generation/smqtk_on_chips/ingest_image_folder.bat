:: Standard process of building IQR-required descriptors+models from scratch
:: and existing configs.
::
:: This assumes the use of the LSH nearest-neighbor index as it builds ITQ model.
::

rem Setup VIAME Paths (no need to set if installed to registry or already set up)

set VIAME_INSTALL=.\..\..\..

call "%VIAME_INSTALL%\setup_viame.bat"

:: PARAMETERS ###################################################################

set IMAGE_FILE_DIR=%1

mkdir models

set IMAGE_INGEST_LIST=models/ingested_image_list.txt

set SMQTK_GEN_IMG_TILES=configs/generate_image_transform.tiles.json

set SMQTK_CMD_CONFIG=configs/compute_many_descriptors.json
set SMQTK_CMD_BATCH_SIZE=1000
set SMQTK_CMD_PROCESSED_CSV=models/alexnet_fc7.cmd.processed.csv

set SMQTK_ITQ_TRAIN_CONFIG=configs/train_itq.json
set SMQTK_ITQ_BIT_SIZE=256

set SMQTK_HCODE_CONFIG=configs/compute_hash_codes.json
set SMQTK_HCODE_PICKLE=models/alexnet_fc7.itq_b256_i50_n2_r0.lsh_hash2uuids.pickle

set SMQTK_BTREE_CONFIG=configs/make_balltree.json

:: Use these tiles for new imagelist
dir %IMAGE_FILE_DIR% /b /s > %IMAGE_INGEST_LIST%

:: Compute descriptors
compute_many_descriptors -v -b %SMQTK_CMD_BATCH_SIZE% --check-image -c "%SMQTK_CMD_CONFIG%" -f "%IMAGE_INGEST_LIST%" -p "%SMQTK_CMD_PROCESSED_CSV%"

:: Train ITQ models
train_itq -vc "%SMQTK_ITQ_TRAIN_CONFIG%"

:: Compute hash codes for descriptors
compute_hash_codes -vc "%SMQTK_HCODE_CONFIG%"

:: Compute balltree hash index
make_balltree -vc "%SMQTK_BTREE_CONFIG%"
