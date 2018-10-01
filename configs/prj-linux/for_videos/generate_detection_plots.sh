#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Processing options
export INPUT_DIRECTORY=videos
export OBJECT_LIST=pristipomoides_auricilla,pristipomoides_zonatus,pristipomoides_sieboldii,etelis_carbunculus,etelis_coruscans,naso,aphareus_rutilans,seriola,hyporthodus_quernus,caranx_melampygus
export FRAME_RATE=5
export DETECTION_THRESHOLD=0.25

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init \
  -d ${INPUT_DIRECTORY} \
  --detection-plots \
  -species ${OBJECT_LIST} \
  -threshold ${DETECTION_THRESHOLD} -frate ${FRAME_RATE} -smooth 2 \
  -p ${VIAME_INSTALL}/configs/pipelines/index_mouss.no_desc.pipe
