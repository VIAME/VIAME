#!/bin/bash

export VIAME_INSTALL=/opt/noaa/viame

# Set these to your maximum image size
export MAX_IMAGE_WIDTH=1400
export MAX_IMAGE_HEIGHT=1000

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init -l input_list.txt \
  --build-index --ball-tree -p pipelines/index_full_frame.res.pipe \
  -archive-width ${MAX_IMAGE_WIDTH} -archive-height ${MAX_IMAGE_HEIGHT} \
  -install ${VIAME_INSTALL}
