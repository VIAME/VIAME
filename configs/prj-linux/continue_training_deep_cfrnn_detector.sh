#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT_DIRECTORY=training_data

# Seed model: the pack written by the training scripts, or the legacy folder
if [ -f trained_model.zip ]; then
  export SEED_MODEL=trained_model.zip
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
    --threshold 0.0 --output-file trained_model.zip
elif [ -d deep_training ]; then
  viame train \
    -i ${INPUT_DIRECTORY} \
    -c ${VIAME_INSTALL}/configs/pipelines/train_detector_netharn_cfrnn.conf \
    --continue \
    --threshold 0.0 --output-file trained_model.zip
else
  echo "Initial model or in progress training folder does not exist, exiting"
fi
