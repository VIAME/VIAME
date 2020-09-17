#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT_DIRECTORY=videos
export OUTPUT_DIRECTORY=output
export OBJECT_LIST=\
pristipomoides_auricilla,\
pristipomoides_zonatus,\
pristipomoides_sieboldii,\
etelis_carbunculus,\
etelis_coruscans,\
naso,\
aphareus_rutilans,\
seriola,\
hyporthodus_quernus,\
caranx_melampygus
export FRAME_RATE=5
export DETECTION_THRESHOLD=0.25

# Extra resource utilization options
export TOTAL_GPU_COUNT=1
export PIPES_PER_GPU=1

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init \
  -d ${INPUT_DIRECTORY} \
  --detection-plots \
  -p ${VIAME_INSTALL}/configs/pipelines/index_mouss.no_desc.pipe \
  -o ${OUTPUT_DIRECTORY} -plot-objects ${OBJECT_LIST} \
  -plot-threshold ${DETECTION_THRESHOLD} -frate ${FRAME_RATE} -plot-smooth 2 \
  -gpus ${TOTAL_GPU_COUNT} -pipes-per-gpu ${PIPES_PER_GPU}
