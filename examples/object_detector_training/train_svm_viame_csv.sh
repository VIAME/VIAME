#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh 

# Perform indexing operation required for SVM model train

python ${VIAME_INSTALL}/configs/process_video.py --init -l input_list.txt \
  --build-index --ball-tree -p pipelines/index_default.svm.pipe \
  -install ${VIAME_INSTALL}

# Perform actual SVM model generation
