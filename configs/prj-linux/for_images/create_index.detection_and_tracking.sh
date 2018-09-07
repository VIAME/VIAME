#!/bin/bash

export VIAME_INSTALL=/opt/noaa/viame

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init -d videos \
  --build-index --ball-tree -p pipelines/ingest_video.tut.pipe \
  -install ${VIAME_INSTALL}
