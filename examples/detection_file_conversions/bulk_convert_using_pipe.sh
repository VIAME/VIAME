 #!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options
export INPUT=../object_detector_training/training_data_mouss
export OUTPUT=example_output
export DEFAULT_FRAME_RATE=5
export PIPELINE=pipelines/convert_viame_csv_to_coco_json.pipe

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py \
  -i ${INPUT} -o ${OUTPUT} -frate ${DEFAULT_FRAME_RATE} \
  -p ${PIPELINE} -auto-detect-gt viame_csv --no-reset-prompt
