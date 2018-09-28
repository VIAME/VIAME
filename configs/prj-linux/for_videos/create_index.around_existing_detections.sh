#!/bin/bash

export VIAME_INSTALL=/opt/noaa/viame

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init \
  -d videos \
  -id input_detections.csv \
  --build-index --ball-tree -p pipelines/index_existing_detections.res.pipe \
  -install ${VIAME_INSTALL}
