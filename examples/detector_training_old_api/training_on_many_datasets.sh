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
  -i ${input_folder}/FalseEx/Filelist.txt  \
  -f habcam -o ${output_folder}/formatted_samples \
  -v ${output_folder}/validation \
  ${common_args}

python format_data_for_training.py \
  -i ${input_folder}/HabCamEx/Groundtruth.txt  \
  -f habcam -o ${output_folder}/formatted_samples \
  -v ${output_folder}/validation \
  --clip-right \
  ${common_args}

# Run Training Module

