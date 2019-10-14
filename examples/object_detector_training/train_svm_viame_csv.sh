#!/bin/sh

# Input locations and types

export INPUT_DIRECTORY=training_data
export ANNOTATION_TYPE=habcam

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh 

# Perform indexing operation required for SVM model train

python ${VIAME_INSTALL}/configs/process_video.py --init -d ${INPUT_DIRECTORY} \
  -p pipelines/index_default.pipe --build-index --ball-tree \
  -auto-detect-gt ${ANNOTATION_TYPE} -install ${VIAME_INSTALL}

# Perform actual SVM model generation
