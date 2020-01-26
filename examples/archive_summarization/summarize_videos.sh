#!/bin/sh

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init -d INPUT_DIRECTORY \
  --detection-plots \
  -plot-objects pristipomoides_auricilla,pristipomoides_zonatus,pristipomoides_sieboldii,etelis_carbunculus,etelis_coruscans,naso,aphareus_rutilans,seriola,hyporthodus_quernus,caranx_melampygus \
  -plot-threshold 0.25 -frate 2 -plot-smooth 2 \
  -p pipelines/index_mouss.no_desc.pipe
