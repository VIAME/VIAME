#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/home/matt/Dev/viame/build/install

# Core processing options
export INPUT="20210627_JACOB ROCK"
export OUTPUT=output

# Extra resource utilization options
export TOTAL_GPU_COUNT=1
export PIPES_PER_GPU=1

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py \
  -i "${INPUT}" -o "${OUTPUT}" -p auto --mosaic \
  -gpus ${TOTAL_GPU_COUNT} -pipes-per-gpu ${PIPES_PER_GPU} \
  -install ${VIAME_INSTALL}
