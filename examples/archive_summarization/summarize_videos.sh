#!/bin/sh

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init -d INPUT_DIRECTORY \
  --detection-plots \
  -plot-threshold 0.25 -frate 2 -plot-smooth 2 \
  -p pipelines/index_default.pipe
