#!/bin/bash

export VIAME_INSTALL=/opt/noaa/viame

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init \
  -l input_list.txt \
  -id input_detections.csv \
  --build-index --ball-tree -p pipelines/ingest_list_and_detections.res.pipe \
  -install ${VIAME_INSTALL}
