#!/bin/bash

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init \
  -l ingest_list.txt -id input_detections.csv \
  -p pipelines/index_existing.pipe -o database \
  --build-index -install ${VIAME_INSTALL}
