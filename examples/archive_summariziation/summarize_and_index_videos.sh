#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL_DIR=./../..
export VIAME_SCRIPT_DIR=${VIAME_INSTALL_DIR}/configs

source ${VIAME_INSTALL_DIR}/setup_viame.sh

python ${VIAME_SCRIPT_DIR}/ingest_video.py --init -d INPUT_DIRECTORY \
  --detection-plots -species fish -threshold 0.25 -frate 0.5 -smooth 2 \
  -p pipelines/ingest_video.mouss_index.pipe --build-index --ball-tree

# Timestamp hack (will be removed in future iterations)

for f in database/*.kw18; do
    awk '{print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18 / 1e6, $19, $20}' < ${f}.bckup > ${f}
done
