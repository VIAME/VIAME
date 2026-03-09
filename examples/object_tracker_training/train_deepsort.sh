#!/bin/sh

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

# Train DeepSORT Re-ID appearance model from groundtruth tracks
#
# DeepSORT trains a neural network to extract appearance features for
# matching detections across frames. GPU required.

viame train \
  -i training_data \
  -tt deepsort \
  --threshold 0.0
