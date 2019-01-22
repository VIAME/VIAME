#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT_DIRECTORY=videos
export FRAME_RATE=5

# Extra resource utilization options
export TOTAL_GPU_COUNT=1
export PIPES_PER_GPU=1

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py \
  -d ${INPUT_DIRECTORY} -frate ${FRAME_RATE} \
  -p pipelines/detector_yolo_default.pipe \
  -gpus ${TOTAL_GPU_COUNT} -pipes-per-gpu ${PIPES_PER_GPU} \
  -s detector:detector:darknet:net_config=deep_training/yolo.cfg \
  -s detector:detector:darknet:weight_file=deep_training/models/yolo.backup \
  -s detector:detector:darknet:class_names=deep_training/yolo.lbl \
  -s detector:detector:darknet:scale=1.4 \
  -s detector_writer:file_name=deep_detections.csv
