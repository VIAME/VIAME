#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options. INPUT is either a single folder of images or a
# multi-camera rig folder containing PORT/STAR/CENTER subfolders (all
# cameras are processed together). Multiple site folders can be given to
# share one coverage grid across them (cross-site / cross-day revisits).
export INPUT=insert_foldername_here
export OUTPUT=output

# Optional GPS metadata: a daily FMCLOG CSV or a directory of them. Leave
# empty to auto-detect an imagelog.json in the image folder or embedded
# EXIF GPS; with no metadata at all, coverage still works within-site from
# the registration chains alone.
export FLIGHT_LOGS=

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Detect previously-observed image regions with the recommended (hybrid)
# method: affine registration chains + rig-constant cross-camera consensus
# for precise recent overlap, and a geo-referenced ground-occupancy grid
# for revisits (later passes, loop closures, earlier sites/days), with GPS
# dead-reckoning across featureless open water. Main output is
# ${OUTPUT}/prior_coverage.csv - a standard VIAME detection CSV with one
# polygon row per previously-seen region for EVERY camera frame, class
# names prior_coverage_sequential / _cross_camera / _revisit. Also writes
# revisits.csv, coverage_map.png and prior_coverage_vis.png.
FLIGHT_LOGS_ARG=""
if [ -n "${FLIGHT_LOGS}" ]; then FLIGHT_LOGS_ARG="--flight-logs ${FLIGHT_LOGS}"; fi

python ${VIAME_INSTALL}/configs/detect_prior_coverage.py "${INPUT}" \
  --method hybrid --output "${OUTPUT}" ${FLIGHT_LOGS_ARG}
