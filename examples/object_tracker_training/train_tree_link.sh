#!/bin/sh

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

# Train the TreeLink tracker from groundtruth tracks
#
# Fits a link scorer and the tracker gates. Add detector output over the
# same clips with -s tracker_trainer:tree_link:computed_detections=FOLDER
# so the scorer sees real detections. No GPU is required.

viame train \
  -i training_data \
  --tracker tree_link \
  --threshold 0.0
