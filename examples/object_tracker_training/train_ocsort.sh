#!/bin/sh

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

# Train OC-SORT tracker parameters from groundtruth tracks
#
# OC-SORT extends ByteTrack with velocity direction consistency.
# No GPU is required.

viame train \
  -i training_data \
  -tt ocsort \
  --threshold 0.0
