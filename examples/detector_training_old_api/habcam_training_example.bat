#!/usr/bin/sh

input_folder=training_data
output_folder=training_output

common_args="-ni 544 -nj 544 --norm --filter" # YOLO (Darknet) settings
#common_args="--no-empty --filter" # FRCNN settings

# Make Output Directory
mkdir -p ${output_folder}

# Extract Training Data in Correct Format
python format_data_for_training.py \
  -i ${input_folder}/HabCamEx/Groundtruth.txt  \
  -f habcam -o ${output_folder}/formatted_samples \
  -v ${output_folder}/validation \
  --clip-right \
  ${common_args}

python format_data_for_training.py \
  -i ${input_folder}/FalseEx/filelist.txt  \
  -f habcam -o ${output_folder}/formatted_samples \
  -v ${output_folder}/validation \
  ${common_args}

# Run Training Module
