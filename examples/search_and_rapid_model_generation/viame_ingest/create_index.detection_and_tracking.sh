#!/bin/bash

export VIAME_INSTALL=./../../..

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init -d videos \
  --build-index --ball-tree -p pipelines/index_default.tut.res.pipe \
  -install ${VIAME_INSTALL}
