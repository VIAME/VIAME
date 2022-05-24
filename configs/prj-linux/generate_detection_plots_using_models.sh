#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT=videos
export OUTPUT=output
export FRAME_RATE=5
export PIPELINE=pipelines/index_mouss.no_desc.pipe
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
export DETECTION_THRESHOLD=0.25

# Extra resource utilization options
export TOTAL_GPU_COUNT=1
export PIPES_PER_GPU=1

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init \
  -i ${INPUT} -o ${OUTPUT} -frate ${FRAME_RATE} \
  -p ${PIPELINE} \
  -plot-objects ${OBJECT_LIST} --detection-plots \
  -plot-threshold ${DETECTION_THRESHOLD} -plot-smooth 2 \
  -gpus ${TOTAL_GPU_COUNT} -pipes-per-gpu ${PIPES_PER_GPU}
