#!/bin/bash

export VIAME_INSTALL=./../../..

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init \
  -l input_list.txt \
  -id input_detections.csv \
  --build-index --ball-tree -p pipelines/index_existing_detections.res.pipe \
  -install ${VIAME_INSTALL}
