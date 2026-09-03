#!/bin/bash

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Report multi-object tracking statistics, per category as well as aggregate.
#
# MOTA, MOTP, IDF1, identity switches, fragmentation and the HOTA family are
# all computed in the same pass as the detection metrics.

viame score \
 --computed detections.csv --truth groundtruth.csv \
 --iou 0.5 --conf 0.10 --per-class \
 --output-summary output_mot_stats_per_category.txt \
 --output-metrics output_mot_stats_per_category.json
