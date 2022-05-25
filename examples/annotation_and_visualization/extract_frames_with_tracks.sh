#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/home/matt/Dev/viame/build/install

# Input and output folders
export INPUT=videos
export OUTPUT=frames

# The default frame rate is only used when the csvs alongside videos
# do not contain frame rates, otherwise the CSV frame rate is used.
export DEFAULT_FRAME_RATE=5

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py \
  -i ${INPUT} -o ${OUTPUT} -frate ${DEFAULT_FRAME_RATE} \
  -p "pipelines/filter_tracks_only.pipe" \
  -auto-detect-gt viame_csv
