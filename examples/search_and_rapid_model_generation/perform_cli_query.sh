#!/bin/bash

# Query the indexed database using tracks from a CSV file
#
# This script takes an input track CSV (containing bounding boxes to query)
# and an image list, queries the database for similar descriptors,
# and outputs results as a track CSV with nearest neighbor scores as confidence.
#
# Usage: ./perform_cli_query.sh
#
# Before running, ensure you have:
#   1. Built an index using one of the create_index.*.sh scripts
#   2. Created an input track CSV (e.g., query_box.csv)
#   3. Created an input image list containing the query image paths
#
# The default files are configured for the mouss example imagery set.

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Default input files
INPUT_TRACKS="${1:-query_box.csv}"
INPUT_LIST="${2:-query_list.txt}"
OUTPUT_FILE="${3:-query_results.csv}"

# Check if input files exist
if [ ! -f "${INPUT_TRACKS}" ]; then
    echo "Error: Input track file not found: ${INPUT_TRACKS}"
    echo "Usage: $0 [input_tracks.csv] [query_list.txt] [output.csv]"
    exit 1
fi

if [ ! -f "${INPUT_LIST}" ]; then
    echo "Error: Input list file not found: ${INPUT_LIST}"
    echo "Usage: $0 [input_tracks.csv] [query_list.txt] [output.csv]"
    exit 1
fi

if [ ! -d "database" ]; then
    echo "Error: Database directory not found. Please run one of the create_index.*.sh scripts first."
    exit 1
fi

PIPELINE_FILE="${VIAME_INSTALL}/configs/pipelines/query_from_track.pipe"
if [ ! -f "${PIPELINE_FILE}" ]; then
    echo "Error: Pipeline file not found: ${PIPELINE_FILE}"
    exit 1
fi

echo ""
echo "============================================"
echo "Perform CLI Query"
echo "============================================"
echo ""
echo "Input tracks:  ${INPUT_TRACKS}"
echo "Input images:  ${INPUT_LIST}"
echo "Output file:   ${OUTPUT_FILE}"
echo ""

# Start the database if not running
python ${VIAME_INSTALL}/configs/database_tool.py start 2>/dev/null || true

# Run the query pipeline
if kwiver runner ${VIAME_INSTALL}/configs/pipelines/query_from_track.pipe \
  -s input:video_filename=${INPUT_LIST} \
  -s track_reader:file_name=${INPUT_TRACKS} \
  -s track_writer:file_name=${OUTPUT_FILE}
then
    if [ -f "${OUTPUT_FILE}" ]; then
        echo ""
        echo "Query complete. Results written to: ${OUTPUT_FILE}"
        echo ""
    else
        echo ""
        echo "Error: Pipeline completed but output file was not created: ${OUTPUT_FILE}"
        echo ""
        exit 1
    fi
else
    echo ""
    echo "Error: Pipeline failed to execute"
    echo ""
    exit 1
fi
