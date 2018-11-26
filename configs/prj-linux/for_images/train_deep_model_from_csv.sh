#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

viame_train_detector \
  -i training_data \
  -c ${VIAME_INSTALL}/configs/pipelines/train_yolo_704.viame_csv.conf \
  --threshold 0.0
