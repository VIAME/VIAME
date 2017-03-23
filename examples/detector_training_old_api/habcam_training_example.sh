#!/usr/bin/sh

# Settings
input_folder=training_data
output_folder=training_output
data_type=".png"
gpu_id=0

common_args="-ni 544 -nj 544 --norm --filter" # YOLO (Darknet) settings
#common_args="--no-empty --filter" # FRCNN settings

source ../../setup_viame.sh

# Make output directory
python create_dir.py -d ${output_folder}

# Extract training data in correct format
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

# Generate input training list and run training
python generate_headers.py -t YOLOv2 \
  -i ${input_folder} \
  -o ${output_folder} \
  -e ${data_type}

darknet -i ${gpu_id} detector train \
  ${output_folder}/YOLOv2.data \
  config_files/YOLOv2.cfg \
  ../detector_pipelines/models/model2.weights
