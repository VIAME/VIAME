#!/bin/sh

export VIAME_INSTALL=/opt/noaa/viame
export VIDEO_DIRECTORY=videos
export SPECIES_LIST=pristipomoides_auricilla,pristipomoides_zonatus,pristipomoides_sieboldii,etelis_carbunculus,etelis_coruscans,naso,aphareus_rutilans,seriola,hyporthodus_quernus,caranx_melampygus

source ${VIAME_INSTALL}/setup_viame.sh

# Generate plots and video ingests
python ${VIAME_INSTALL}/configs/ingest_video.py --init -d ${VIDEO_DIRECTORY} \
  --detection-plots \
  -species ${SPECIES_LIST} \
  -threshold 0.25 -frate 2 -smooth 2 \
  -p ${VIAME_INSTALL}/configs/pipelines/index_mouss.no_desc.pipe
