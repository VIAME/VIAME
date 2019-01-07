@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_LIST=input_list.txt
SET INPUT_FRAME_RATE=1
SET PROCESS_FRAME_RATE=1

REM Note: Frame rates are specified in hertz, aka frames per second. If the
REM input frame rate is 1 and the process frame rate is also 1, then every
REM input image in the list will be processed. If the process frame rate
REM is 2, then every other image will be processed.

REM Extra resource utilization options
SET TOTAL_GPU_COUNT=1
SET PIPES_PER_GPU=1

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" ^
  -l "%INPUT_LIST%" -ifrate %INPUT_FRAME_RATE% -frate %PROCESS_FRAME_RATE% ^
  -p pipelines\detector_yolo_default.pipe ^
  -gpus %TOTAL_GPU_COUNT% -pipes-per-gpu %PIPES_PER_GPU% ^
  -s detector:detector:darknet:net_config=deep_training\yolo_v2.cfg ^
  -s detector:detector:darknet:weight_file=deep_training\models\yolo_v2.backup ^
  -s detector:detector:darknet:class_names=deep_training\yolo_v2.lbl ^
  -s detector:detector:darknet:scale=1.4 ^
  -s detector_writer:file_name=deep_detections.csv

pause
