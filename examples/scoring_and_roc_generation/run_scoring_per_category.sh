#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh

# Run score tracks on data for singular metrics

python ${VIAME_INSTALL}/configs/score_results.py \
 -computed detections.csv -truth groundtruth.csv \
 -threshold 0.05 -stats score_tracks_output.txt \
 --per-category

# Generate ROC

python ${VIAME_INSTALL}/configs/score_results.py \
 -computed detections.csv -truth groundtruth.csv \
 -threshold 0.05 -roc roc.png \
 --per-category

