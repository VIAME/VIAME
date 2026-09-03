#!/bin/bash

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Write precision-recall, ROC and confusion-matrix data, reporting each
# category separately in addition to the aggregate.
#
# The curves are emitted as CSV alongside rendered plots, so they can be
# replotted or diffed without rerunning the scoring.

viame score \
 --computed detections.csv --truth groundtruth.csv \
 --iou 0.5 --per-class \
 --output-pr-csv output_prc_and_conf_mat_per_class/pr_curve.csv \
 --output-roc-csv output_prc_and_conf_mat_per_class/roc_curve.csv \
 --output-conf-csv output_prc_and_conf_mat_per_class/confusion_matrix.csv \
 --output-plots output_prc_and_conf_mat_per_class
