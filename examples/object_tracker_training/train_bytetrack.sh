#!/bin/sh

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

# Train ByteTrack tracker parameters from groundtruth tracks
#
# ByteTrack training estimates optimal Kalman filter parameters from
# groundtruth track data. No GPU is required.

viame train \
  -i training_data \
  -tt bytetrack \
  --threshold 0.0
