#!/bin/sh

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init -d INPUT_DIRECTORY \
  --detection-plots \
  -objects pristipomoides_auricilla,pristipomoides_zonatus,pristipomoides_sieboldii,etelis_carbunculus,etelis_coruscans,naso,aphareus_rutilans,seriola,hyporthodus_quernus,caranx_melampygus \
  -plot-threshold 0.25 -frate 2 -smooth 2 \
  -p pipelines/index_mouss.pipe --build-index --ball-tree
