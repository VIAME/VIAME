#!/bin/bash

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Generate ROC

python ${VIAME_INSTALL}/configs/score_results.py \
 -computed detections.csv -truth groundtruth.csv \
 -det-prc-conf conf_mat_per_category.png --per-category
