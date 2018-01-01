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

SMQTK_GEN_IMG_TILES="configs/generate_image_transform.tiles.json"

# Compute tiles via chipping up the input images in a grid-like pattern
if [ -n "$(which parallel 2>/dev/null)" ]
then
    cat "${IMAGE_DIR_FILELIST}" | parallel "
        generate_image_transform -c \"${SMQTK_GEN_IMG_TILES}\" \
            -i \"{}\" -o \"${IMAGE_TILES_DIR}\"
    "
else
    cat "${IMAGE_DIR_FILELIST}" | \
        xargs -I '{}' generate_image_transform -c "${SMQTK_GEN_IMG_TILES}" \
           -i '{}' -o "${IMAGE_TILES_DIR}"
fi

# Ingest descriptors around each chip
#bash ingest_chip_folder.sh ${IMAGE_TILES_DIR}
