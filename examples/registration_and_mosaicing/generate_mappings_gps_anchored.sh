#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options
export INPUT=insert_foldername_here
export OUTPUT=output

# Optional GPS metadata. Leave FLIGHT_LOG empty to auto-detect an imagelog.json
# in the image folder or embedded EXIF GPS; otherwise point it at an FMCLOG CSV.
export FLIGHT_LOG=

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Same recommended sequential-registration settings, plus --geo-anchor, which
# fits a GLOBAL GPS-to-pixel transform: it places featureless water frames by
# dead-reckoning and reports how far the sequential feature chain has drifted
# from the GPS truth. The flight-log flag is optional when an imagelog.json /
# EXIF GPS is present.
FLIGHT_LOG_ARG=""
if [ -n "${FLIGHT_LOG}" ]; then FLIGHT_LOG_ARG="--flight-log ${FLIGHT_LOG}"; fi

python ${VIAME_INSTALL}/configs/reconstruct_3d.py "${INPUT}" \
  --output "${OUTPUT}" \
  --planar --coverage-class suppressed --visualize \
  --affine --consistency-filter --xcam-robust --xcam-low-drift \
  --geo-anchor ${FLIGHT_LOG_ARG}
