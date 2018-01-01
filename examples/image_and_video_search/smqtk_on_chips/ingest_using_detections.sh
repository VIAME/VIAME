#!/usr/bin/env bash
#
# Standard process of building IQR-required descriptors+models from scratch
# and existing configs.
#
# This assumes the use of the LSH nearest-neighbor index as it builds ITQ model.
#
set -e

source ../../../setup_viame.sh

# PARAMETERS ###################################################################

IMAGE_LIST="input_list.txt"
IMAGE_TILES_DIR="tiles"

# Compute tiles using KWIVER pipeline
echo "Generating tiles for images ($(wc -l "${IMAGE_LIST}" | cut -d' ' -f1) images)"
mkdir -p "${IMAGE_TILES_DIR}"
pipeline_runner -p configs/chip_extractor_pipeline.pipe
