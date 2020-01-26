#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh 

# Run pipeline

viame_train_detector \
  -i training_data_habcam \
  -c ${VIAME_INSTALL}/configs/pipelines/train_mmdet_cascade.habcam.conf \
  --threshold 0.0
