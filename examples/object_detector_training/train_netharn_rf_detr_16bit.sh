#!/bin/sh

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

# Train RF-DETR detector on 16-bit imagery with automatic normalization
# The --normalize-16bit flag enables percentile normalization for non-8-bit imagery
viame train \
  --input-list input_list_arctic_seal_16bit.txt \
  --input-truth groundtruth_arctic_seal_16bit.csv \
  --labels labels_arctic_seal_16bit.txt \
  -c ${VIAME_INSTALL}/configs/pipelines/train_detector_netharn_rf_detr.conf \
  --normalize-16bit \
  --threshold 0.0
