#!/bin/sh

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

# Train tracker using adaptive selection
#
# The adaptive trainer analyzes tracking data statistics (track counts,
# lengths, motion patterns, density, occlusion) and automatically selects
# up to 3 trackers to train: ByteTrack, OC-SORT, DeepSORT, BoT-SORT, SRNN.

viame train \
  -i training_data \
  -c ${VIAME_INSTALL}/configs/pipelines/train_tracker_adaptive.conf \
  --threshold 0.0
