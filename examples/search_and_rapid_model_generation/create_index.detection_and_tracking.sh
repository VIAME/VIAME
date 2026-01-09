#!/bin/bash

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# To change this script to process a directory of videos, as opposed to images
# change "-l ingest_list.txt" to "-d videos" if videos is a directory with videos

python ${VIAME_INSTALL}/configs/process_video.py --init \
  -l ingest_list.txt \
  -p pipelines/index_default.trk.pipe -o database \
  --build-index -install ${VIAME_INSTALL}
