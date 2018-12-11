#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh 

# Run pipeline

viame_train_detector \
  -i training_data_habcam \
  -c ${VIAME_INSTALL}/configs/pipelines/train_yolo_704.habcam.conf \
  -p ${VIAME_INSTALL}/configs/pipelines/training_split_and_stereo.pipe \
  --threshold 0.0
