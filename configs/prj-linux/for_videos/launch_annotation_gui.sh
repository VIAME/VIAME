#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Processing options
export INPUT_DIRECTORY=videos
export CACHE_DIRECTORY=database
export FRAME_RATE=5

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/launch_annotation_gui.py \
  -d ${INPUT_DIRECTORY} -c ${CACHE_DIRECTORY} -frate ${FRAME_RATE} 
