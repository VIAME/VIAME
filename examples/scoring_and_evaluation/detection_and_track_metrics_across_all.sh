#!/bin/bash

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Score detections and tracks against groundtruth, treating all categories as one.
#
# Writes a metric summary, a metrics json, and plots (precision-recall curve,
# detection ROC curve, confusion matrix, and score histograms) into output_metrics.

viame_score_results \
 --computed detections.csv --truth groundtruth.csv \
 --iou 0.5 \
 --output-summary output_metrics_summary.txt \
 --output-metrics output_metrics.json \
 --output-plots output_metrics
