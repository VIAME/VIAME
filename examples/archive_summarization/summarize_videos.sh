#!/bin/sh

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

viame run --init -d INPUT_DIRECTORY \
  --detection-plots \
  -plot-threshold 0.25 -frate 2 -plot-smooth 2 \
  -p pipelines/index_generic.pipe
