#!/bin/bash

export VIAME_INSTALL=./../../..

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init -l input_list.txt \
  --build-index --ball-tree -p pipelines/index_full_frame.pipe \
  -archive-width 1400 -archive-height 1000 \
  -install ${VIAME_INSTALL}
