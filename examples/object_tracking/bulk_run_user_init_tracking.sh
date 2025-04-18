 #!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options
export INPUT_FOLDER=../object_detector_training/training_data_mouss
export INPUT_FORMAT=viame_csv
export OUTPUT_FOLDER=example_output
export DEFAULT_FRAME_RATE=5

# Setup paths, pipeline, and run the command
export PIPELINE=pipelines/utility_track_selections_default_mask.pipe

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py \
  -i ${INPUT_FOLDER} -o ${OUTPUT_FOLDER} -frate ${DEFAULT_FRAME_RATE} \
  -p ${PIPELINE} -auto-detect-gt ${INPUT_FORMAT} --no-reset-prompt
