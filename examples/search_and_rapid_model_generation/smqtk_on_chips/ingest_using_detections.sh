#!/usr/bin/env bash
#
# Standard process of building IQR-required descriptors+models from scratch
# and existing configs.
#
# This assumes the use of the LSH nearest-neighbor index as it builds ITQ model.
#
set -e

export VIAME_INSTALL=./../../..

source ${VIAME_INSTALL}/setup_viame.sh 

# PARAMETERS ###################################################################

IMAGE_LIST="input_list.txt"
IMAGE_TILES_DIR="tiles"

# Compute tiles using KWIVER pipeline
echo "Generating tiles for images ($(wc -l "${IMAGE_LIST}" | cut -d' ' -f1) images)"
mkdir -p "${IMAGE_TILES_DIR}"

kwiver runner ${VIAME_INSTALL}/configs/pipelines/detector_extract_chips.pipe \
                -s input:video_filename=${IMAGE_LIST}

# Perform ingest on computed chips
bash ingest_image_folder.sh "${IMAGE_TILES_DIR}"
