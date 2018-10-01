#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Processing options
export INPUT_DIRECTORY=videos
export FRAME_RATE=5

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

pipeline_runner -p ${VIAME_INSTALL}/configs/pipelines/tracker_default.tut.pipe \
                -s input:video_filename=input_list.txt \
                -s detector:detector:darknet:net_config=deep_training/yolo_v2.cfg \
                -s detector:detector:darknet:weight_file=deep_training/models/yolo_v2.backup \
                -s detector:detector:darknet:class_names=deep_training/yolo_v2.lbl \
                -s detector:detector:darknet:scale=1.4 \
                -s detector_writer:file_name=deep_detections.csv \
                -s track_writer:file_name=deep_tracks.csv
