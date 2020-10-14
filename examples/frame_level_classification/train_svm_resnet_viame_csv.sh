#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options
export INPUT_DIRECTORY=training_data

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info

viame_train_detector \
  -i ${INPUT_DIRECTORY} \
  -c ${VIAME_INSTALL}/configs/pipelines/train_svm_full_frame_classifier.viame_csv.conf \
  --threshold 0.0
