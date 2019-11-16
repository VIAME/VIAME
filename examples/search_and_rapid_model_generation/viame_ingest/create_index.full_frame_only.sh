#!/bin/bash

export VIAME_INSTALL=./../../..

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init -l input_list.txt \
  --build-index -p pipelines/index_full_frame.pipe \
  -install ${VIAME_INSTALL}
