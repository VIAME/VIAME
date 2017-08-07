#!/usr/bin/env bash
#
# Standard process of building IQR-required descriptors+models from scratch
# and existing configs.
#
# This assumes the use of the LSH nearest-neighbor index as it builds ITQ model.
#
set -e

# PARAMETERS ###################################################################

IMAGE_TILES_DIR=$1

SMQTK_GEN_IMG_TILES="configs/generate_image_transform.tiles.json"

SMQTK_CMD_CONFIG="configs/compute_many_descriptors.json"
SMQTK_CMD_BATCH_SIZE=1000
SMQTK_CMD_PROCESSED_CSV="models/alexnet_fc7.cmd.processed.csv"

SMQTK_ITQ_TRAIN_CONFIG="configs/train_itq.json"
SMQTK_ITQ_BIT_SIZE=256

SMQTK_HCODE_CONFIG="configs/compute_hash_codes.json"
SMQTK_HCODE_PICKLE="models/alexnet_fc7.itq_b256_i50_n2_r0.lsh_hash2uuids.pickle"

SMQTK_HCODE_BTREE_LEAFSIZE=40
SMQTK_HCODE_BTREE_RAND=0
SMQTK_HCODE_BTREE_OUTPUT="models/alexnet_fc7.itq_b256_i50_n2_r0.hi_btree.npz"

# Use these tiles for new imagelist
mv "${IMAGE_DIR_FILELIST}" "${IMAGE_DIR_FILELIST}.ORIG"
find "${IMAGE_TILES_DIR}" -type f >"${IMAGE_DIR_FILELIST}"

# Compute descriptors
compute_many_descriptors \
    -v -b ${SMQTK_CMD_BATCH_SIZE} --check-image -c "${SMQTK_CMD_CONFIG}" \
    -f "${IMAGE_DIR_FILELIST}" -p "${SMQTK_CMD_PROCESSED_CSV}"

# Train ITQ models
train_itq -vc "${SMQTK_ITQ_TRAIN_CONFIG}"

# Compute hash codes for descriptors
compute_hash_codes \
    -vc "${SMQTK_HCODE_CONFIG}" \
    --output-hash2uuids "${SMQTK_HCODE_PICKLE}"

# Compute balltree hash index
make_balltree "${SMQTK_HCODE_PICKLE}" ${SMQTK_ITQ_BIT_SIZE} \
    ${SMQTK_HCODE_BTREE_LEAFSIZE} ${SMQTK_HCODE_BTREE_RAND} \
    ${SMQTK_HCODE_BTREE_OUTPUT}
