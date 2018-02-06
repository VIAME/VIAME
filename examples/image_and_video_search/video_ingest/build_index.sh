#!/bin/bash

source ../../../setup_viame.sh

mkdir -p database/ITQ

SMQTK_ITQ_TRAIN_CONFIG="configs/smqtk_train_itq.json"

SMQTK_HCODE_CONFIG="configs/smqtk_compute_hashes.json"
SMQTK_HCODE_PICKLE="database/ITQ/alexnet_fc7.itq_b256_i50_n2_r0.lsh_hash2uuids.pickle"

SMQTK_BTREE_CONFIG="configs/smqtk_make_balltree.json"

# Train ITQ models on ingested descriptors
echo "1. Training ITQ Model"
train_itq -vc "${SMQTK_ITQ_TRAIN_CONFIG}"

# Compute hash codes for descriptors
echo "2. Computing Hash Codes"
compute_hash_codes -vc "${SMQTK_HCODE_CONFIG}"

# Compute balltree hash index
echo "3. Generating Ball Tree"
make_balltree -vc "${SMQTK_BTREE_CONFIG}"
