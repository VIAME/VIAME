#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT_DIRECTORY=training_data
export INITIAL_MODEL=category_models/trained_detector.zip

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

if [ -f ${INITIAL_MODEL} ]; then
  viame_train_detector \
    -i ${INPUT_DIRECTORY} \
    -c ${VIAME_INSTALL}/configs/pipelines/train_netharn_cascade.viame_csv.conf \
    -s detector_trainer:ocv_windowed:trainer:netharn:seed_model=${INITIAL_MODEL} \
    --threshold 0.0
elif [ -d deep_training ]; then
  viame_train_detector \
    -i ${INPUT_DIRECTORY} \
    -c ${VIAME_INSTALL}/configs/pipelines/train_netharn_cascade_nf.viame_csv.conf \
    -s detector_trainer:ocv_windowed:skip_format=true \
    --threshold 0.0
else
  echo "Initial model or in progress training folder does not exist, exiting"
fi
