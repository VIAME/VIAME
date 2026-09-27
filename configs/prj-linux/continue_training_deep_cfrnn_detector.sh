#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT_DIRECTORY=training_data

# Seed model: the pack written by training, or an unpacked folder
if [ -f trained_model.zip ]; then
  export SEED_MODEL=trained_model.zip
elif [ -d trained_model ]; then
  export SEED_MODEL=trained_model
else
  export SEED_MODEL=category_models/trained_detector.zip
fi

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

if [ -f ${SEED_MODEL} ]; then
  viame train \
    -i ${INPUT_DIRECTORY} \
    -c ${VIAME_INSTALL}/configs/pipelines/train_detector_netharn_cfrnn.conf \
    --init-weights ${SEED_MODEL} \
    --threshold 0.0
elif [ -d deep_training ]; then
  viame train \
    -i ${INPUT_DIRECTORY} \
    -c ${VIAME_INSTALL}/configs/pipelines/train_detector_netharn_cfrnn.conf \
    --continue \
    --threshold 0.0
else
  echo "Initial model or in progress training folder does not exist, exiting"
fi
