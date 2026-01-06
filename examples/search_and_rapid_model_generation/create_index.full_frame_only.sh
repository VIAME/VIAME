#!/bin/bash

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init \
  -l ingest_list.txt \
  -p pipelines/index_frame.pipe -o database \
  --build-index -install ${VIAME_INSTALL}
