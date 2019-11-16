#!/bin/bash

export VIAME_INSTALL=./../../..

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init \
  -l input_list.txt \
  -id input_detections.csv \
  --build-index -p pipelines/index_existing.pipe \
  -install ${VIAME_INSTALL}
