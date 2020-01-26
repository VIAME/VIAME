#!/usr/bin/env bash
#
# Standard process of building IQR-required descriptors+models from scratch
# and existing configs.
#
# This assumes the use of the LSH nearest-neighbor index as it builds ITQ model.
#
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../../.."

source ${VIAME_INSTALL}/setup_viame.sh 

# PARAMETERS ###################################################################

IMAGE_FILE_DIR=$1

mkdir -p "models"

IMAGE_INGEST_LIST="models/ingested_image_list.txt"

SMQTK_GEN_IMG_TILES="configs/generate_image_transform.tiles.json"

SMQTK_CMD_CONFIG="configs/compute_many_descriptors.json"
SMQTK_CMD_BATCH_SIZE=1000
SMQTK_CMD_PROCESSED_CSV="models/alexnet_fc7.cmd.processed.csv"

SMQTK_ITQ_TRAIN_CONFIG="configs/train_itq.json"
SMQTK_ITQ_BIT_SIZE=256

SMQTK_HCODE_CONFIG="configs/compute_hash_codes.json"
SMQTK_HCODE_PICKLE="models/alexnet_fc7.itq_b256_i50_n2_r0.lsh_hash2uuids.pickle"

SMQTK_BTREE_CONFIG="configs/make_balltree.json"

# Use these tiles for new imagelist
find "${IMAGE_FILE_DIR}" -type f >"${IMAGE_INGEST_LIST}"

# Compute descriptors
compute_many_descriptors \
    -v -b ${SMQTK_CMD_BATCH_SIZE} --check-image -c "${SMQTK_CMD_CONFIG}" \
    -f "${IMAGE_INGEST_LIST}" -p "${SMQTK_CMD_PROCESSED_CSV}"

# Train ITQ models
train_itq -vc "${SMQTK_ITQ_TRAIN_CONFIG}"

# Compute hash codes for descriptors
compute_hash_codes \
    -vc "${SMQTK_HCODE_CONFIG}"

# Compute balltree hash index
make_balltree \
    -vc "${SMQTK_BTREE_CONFIG}"
