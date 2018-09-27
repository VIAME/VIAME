#!/bin/bash

export VIAME_INSTALL=./../../..

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init -l input_list.txt \
  --build-index --ball-tree -p pipelines/index_default.res.pipe \
  -install ${VIAME_INSTALL}
