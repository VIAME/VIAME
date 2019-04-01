#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh 

# Run pipeline

viame_train_detector \
  -i training_data_mouss \
  -c ${VIAME_INSTALL}/configs/pipelines/train_yolo_wtf_704.kw18.conf \
  --threshold 0.0
