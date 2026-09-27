#!/bin/sh

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

# Train ByteTrack, or a registration-based tracker when the groundtruth
# clearly needs one (see README)

viame train \
  -i training_data \
  -c ${VIAME_INSTALL}/configs/pipelines/train_tracker_default.conf \
  --threshold 0.0
