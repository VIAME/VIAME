#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Processing options
export INPUT_DIRECTORY=videos
export FRAME_RATE=5

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py \
  -d ${INPUT_DIRECTORY} -frate ${FRAME_RATE} \
  -p pipelines/detector_yolo_default.pipe \
  -s detector:detector:darknet:net_config=deep_training/yolo_v2.cfg \
  -s detector:detector:darknet:weight_file=deep_training/models/yolo_v2.backup \
  -s detector:detector:darknet:class_names=deep_training/yolo_v2.lbl \
  -s detector:detector:darknet:scale=1.4 \
  -s detector_writer:file_name=deep_detections.csv
