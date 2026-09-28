#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT=videos
export OUTPUT=output
export FRAME_RATE=5

# Extra resource utilization options
export TOTAL_GPU_COUNT=1
export PIPES_PER_GPU=1

# Trained model: the pack written by training, or an unpacked folder
if [ -f trained_model.zip ]; then
  export TRAINED_MODEL=trained_model.zip/detector.pipe
elif [ -d trained_model ]; then
  export TRAINED_MODEL=trained_model/detector.pipe
else
  export TRAINED_MODEL=category_models/detector.pipe
fi

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Set current directory for project folder pipe
export VIAME_PROJECT_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)"

viame run \
  -i ${INPUT} -o ${OUTPUT} -frate ${FRAME_RATE} \
  -p ${TRAINED_MODEL} --no-reset-prompt \
  -gpus ${TOTAL_GPU_COUNT} -pipes-per-gpu ${PIPES_PER_GPU}
