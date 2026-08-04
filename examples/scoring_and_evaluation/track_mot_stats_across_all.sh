#!/bin/bash

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Report multi-object tracking statistics, treating all categories as one.
#
# MOTA, MOTP, IDF1, identity switches, fragmentation and the HOTA family are
# all computed in the same pass as the detection metrics.

viame_score_results \
 --computed detections.csv --truth groundtruth.csv \
 --iou 0.5 --conf 0.10 \
 --output-summary output_mot_stats.txt \
 --output-metrics output_mot_stats.json
