#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Processing options
export INPUT_DIRECTORY=videos
export FRAME_RATE=5
export START_TIME=00:00:00.00
export OUTPUT_DIR=frames

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

mkdir -p ${OUTPUT_DIR}

for file in ${INPUT_DIRECTORY}/*
do
  file_no_path=${file##*/}
  output_folder=${OUTPUT_DIR}/${file_no_path}
  mkdir -p ${output_folder}

  ffmpeg -i ${file} -r ${FRAME_RATE} \
    -ss ${START_TIME} \
    ${output_folder}/frame%06d.png
done
