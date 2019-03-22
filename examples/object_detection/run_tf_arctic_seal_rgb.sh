#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=../../../build/install

source ${VIAME_INSTALL}/setup_viame.sh 

# Run pipeline

pipeline_runner -p seal_rgb_tf_detector.pipe
