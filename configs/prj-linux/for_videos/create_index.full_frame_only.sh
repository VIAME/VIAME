#!/bin/bash

export VIAME_INSTALL=/opt/noaa/viame

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init -l input_list.txt \
  --build-index --ball-tree -p pipelines/index_full_frame.res.pipe \
  -archive-width 1400 -archive-height 1000 \
  -install ${VIAME_INSTALL}
