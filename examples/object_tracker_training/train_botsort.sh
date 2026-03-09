#!/bin/sh

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

# Train BoT-SORT Re-ID model with camera motion compensation
#
# BoT-SORT combines Re-ID model training with camera motion compensation.
# Best for moving camera scenarios (drones, ROVs, handheld). GPU required.

viame train \
  -i training_data \
  -tt botsort \
  --threshold 0.0
