#!/bin/bash

export VIAME_INSTALL=/opt/noaa/viame

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init -l input_list.txt \
  --build-index --ball-tree -p pipelines/index_default.res.pipe \
  -install ${VIAME_INSTALL}
