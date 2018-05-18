@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

CALL .\..\..\setup_viame.bat

REM Setup Settings

SET input_folder=training_data
SET output_folder=training_output
SET data_type=".png"
SET gpu_id=0

REM YOLO (Darknet) settings
REM SET common_args=-ni 544 -nj 544 --norm --filter

REM FRCNN settings
REM SET common_args=--no-empty --filter

REM Make output directory
python.exe create_dir.py -d %output_folder%

REM Extract training data in correct format
python.exe format_data_for_training.py ^
  -i %input_folder%/HabCamEx/Groundtruth.txt  ^
  -f habcam -o %output_folder%/formatted_samples ^
  -v %output_folder%/validation ^
  -ni 544 -nj 544 --norm --filter

python.exe format_data_for_training.py ^
  -i %input_folder%/FalseEx/filelist.txt  ^
  -f habcam -o %output_folder%/formatted_samples ^
  -v %output_folder%/validation ^
  -ni 544 -nj 544 --norm --filter

REM Generate input training list and run training
python.exe generate_headers.py -t YOLOv2 ^
  -i %input_folder% ^
  -o %output_folder% ^
  -e %data_type%

%VIAME_INSTALL%\bin\darknet.exe -i %gpu_id% detector train ^
  %output_folder%/YOLOv2.data ^
  config_files/YOLOv2.cfg ^
  ../detector_pipelines/models/model2.weights

pause