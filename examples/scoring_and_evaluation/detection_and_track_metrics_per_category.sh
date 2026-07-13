#!/bin/bash

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Score detections and tracks against groundtruth, reporting each category
# separately in addition to the aggregate scores.
#
# The per-class table reports TP, FP, FN, precision, recall, F1 and average
# precision for every category, and the summary reports their mean AP.

viame_score_results \
 --computed detections.csv --truth groundtruth.csv \
 --iou 0.5 --per-class \
 --output-summary output_metrics_per_category_summary.txt \
 --output-metrics output_metrics_per_category.json \
 --output-plots output_metrics_per_category
