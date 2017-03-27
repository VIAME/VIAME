#!/usr/bin/sh

input_folder=training_data
output_folder=training_output

common_args="-ni 544 -nj 544 --norm --filter" # YOLO (Darknet) settings
#common_args="--no-empty --filter" # FRCNN settings

# Make Output Directory
mkdir -p ${output_folder}

# Extract Training Data in Correct Format
python format_data_for_training.py \
  -i ${input_folder}/LabeledFishesInTheWild/Positive/GroundTruth.dat \
  -f wild -o ${output_folder}/formatted_samples \
  -v ${output_folder}/validation \
  ${common_args}

python format_data_for_training.py \
  -i ${input_folder}/FishCLEF15/video1/cadb2ec9\#201102031130_s3_3.xml \
  -f clef -o ${output_folder}/formatted_samples \
  -v ${output_folder}/validation \
  -s clef1_ -e ${input_folder}/clef_exclude.txt \
  ${common_args}

python format_data_for_training.py \
  -i ${input_folder}/FishCLEF15/video2/cadb2ec9\#201105051700_0.xml \
  -f clef -o ${output_folder}/formatted_samples \
  -v ${output_folder}/validation  \
  -s clef2_ -e ${input_folder}/clef_exclude.txt \
  ${common_args}

python format_data_for_training.py \
  -i ${input_folder}/FalseEx/filelist.txt  \
  -f habcam -o ${output_folder}/formatted_samples \
  -v ${output_folder}/validation \
  ${common_args}

python format_data_for_training.py \
  -i ${input_folder}/HabCamEx/Groundtruth.txt  \
  -f habcam -o ${output_folder}/formatted_samples \
  -v ${output_folder}/validation \
  --clip-right \
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
