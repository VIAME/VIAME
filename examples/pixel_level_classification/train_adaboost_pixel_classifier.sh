#!/bin/sh
# Input locations and types
export INPUT_DIRECTORY=training_data_adaboost

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname $0)" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run pipeline
viame_train_detector \
  -i ${INPUT_DIRECTORY} \
  -c ${VIAME_INSTALL}/configs/pipelines/train_adaboost_pixel_classifier.viame_csv.conf \
  --threshold 0.0
