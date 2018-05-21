#!/bin/bash

export VIAME_INSTALL_DIR=./../../..
export VIAME_SCRIPT_DIR=${VIAME_INSTALL_DIR}/configs

source ${VIAME_INSTALL_DIR}/setup_viame.sh

python ${VIAME_SCRIPT_DIR}/ingest_video.py --init -l input_list.txt \
  --build-index --ball-tree -p pipelines/ingest_list.res.pipe
