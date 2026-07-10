#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options
export INPUT=insert_foldername_here
export OUTPUT=output

# Optional GPS metadata: a daily FMCLOG CSV or a directory of them (an
# imagelog.json or EXIF GPS in the input folder is auto-detected if left
# empty; with no metadata at all, revisits are still detected within-site
# from the registration chains alone).
export FLIGHT_LOGS=

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Identify loop-closure / revisit events (the platform leaves a location and
# later returns to image the same ground). Revisit detection lives in
# detect_prior_coverage.py: a geo-referenced ground-occupancy grid flags
# frames that re-cover previously seen ground, and land-to-land events are
# confirmed by direct image registration. Writes revisits.csv and
# coverage_map.png into ${OUTPUT}. Use "--method metadata" for a fast
# GPS-only pass, or drop --revisits-only to also get the full per-frame
# prior-coverage polygons.
FLIGHT_LOGS_ARG=""
if [ -n "${FLIGHT_LOGS}" ]; then FLIGHT_LOGS_ARG="--flight-logs ${FLIGHT_LOGS}"; fi

python ${VIAME_INSTALL}/configs/detect_prior_coverage.py "${INPUT}" \
  --method hybrid --revisits-only --output "${OUTPUT}" ${FLIGHT_LOGS_ARG}
