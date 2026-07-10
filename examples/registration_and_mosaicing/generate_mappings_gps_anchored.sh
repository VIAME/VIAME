#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options
export INPUT=insert_foldername_here
export OUTPUT=output

# Optional GPS metadata: a daily FMCLOG CSV or a directory of them. Leave
# empty to auto-detect an imagelog.json in the image folder or embedded
# EXIF GPS.
export FLIGHT_LOGS=

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Sequential registration + prior coverage WITH GPS anchoring.
# detect_prior_coverage.py calibrates a metres-to-pixels map from the raw
# pairwise registrations (bounded by the altitude/focal-length expectation),
# places featureless open-water frames by GPS dead-reckoning, and tracks all
# observed ground in a geo-referenced occupancy grid so revisits — later
# passes, loop closures, or (multi-folder runs) earlier sites/days — are
# detected and confirmed by direct registration. Writes prior_coverage.csv
# (polygon classes prior_coverage_sequential / _cross_camera / _revisit),
# revisits.csv, coverage_map.png and a thumbnail visualization into
# ${OUTPUT}.
FLIGHT_LOGS_ARG=""
if [ -n "${FLIGHT_LOGS}" ]; then FLIGHT_LOGS_ARG="--flight-logs ${FLIGHT_LOGS}"; fi

python ${VIAME_INSTALL}/configs/detect_prior_coverage.py "${INPUT}" \
  --method hybrid --output "${OUTPUT}" ${FLIGHT_LOGS_ARG}
