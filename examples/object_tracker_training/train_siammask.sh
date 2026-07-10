#!/bin/sh

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

# Train SiamMask visual tracking network
#
# SiamMask training learns to match object templates across frames,
# producing both bounding boxes and segmentation masks. GPU required.

viame train \
  -i training_data \
  -tt siammask \
  --threshold 0.0
