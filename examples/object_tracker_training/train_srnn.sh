#!/bin/sh

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

# Train SRNN multi-stage tracking model
#
# SRNN training is a multi-stage process: Siamese model training, feature
# extraction, individual LSTM training, and combined SRNN training.
# Requires substantial data (100+ annotated tracks). GPU required.

viame train \
  -i training_data \
  -tt srnn \
  --threshold 0.0
