#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options
export INPUT=insert_foldername_here
export OUTPUT=output

# Optional GPS metadata for the metadata method (auto-detects imagelog.json /
# EXIF GPS if left empty; otherwise point at an FMCLOG CSV).
export FLIGHT_LOG=

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Identify loop-closure / revisit events (the platform leaves a location and
# later returns to image the same ground). --method both runs the GPS-metadata
# detector and the image-registration detector and cross-references them.
mkdir -p "${OUTPUT}"
FLIGHT_LOG_ARG=""
if [ -n "${FLIGHT_LOG}" ]; then FLIGHT_LOG_ARG="--flight-log ${FLIGHT_LOG}"; fi

python ${VIAME_INSTALL}/configs/detect_site_revisits.py "${INPUT}" \
  --method both --output "${OUTPUT}/site_revisits.csv" ${FLIGHT_LOG_ARG}
