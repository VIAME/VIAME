#!/bin/bash

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run score tracks on data for singular metrics

python ${VIAME_INSTALL}/configs/score_results.py \
 -computed detections.csv -truth groundtruth.csv \
 -threshold 0.10 -trk-mot-stats output_mot_stats.txt
