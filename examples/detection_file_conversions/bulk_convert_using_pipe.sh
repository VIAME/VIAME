 #!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options
export INPUT_FOLDER=../object_detector_training/training_data_mouss
export INPUT_FORMAT=viame_csv
export OUTPUT_FOLDER=example_output
export OUTPUT_FORMAT=coco_json
export OUTPUT_EXTENSION=json
export DEFAULT_FRAME_RATE=5

# Setup paths, pipeline, and run the command
export PIPELINE=pipelines/convert_${INPUT_FORMAT}_to_${OUTPUT_FORMAT}.pipe

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py \
  -i ${INPUT_FOLDER} -o ${OUTPUT_FOLDER} -frate ${DEFAULT_FRAME_RATE} \
  -p ${PIPELINE} -output-ext ${OUTPUT_EXTENSION} \
  -auto-detect-gt ${INPUT_FORMAT} --no-reset-prompt
