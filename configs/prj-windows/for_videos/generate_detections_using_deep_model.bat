@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=videos
SET FRAME_RATE=5

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\ingest_video.py" ^
  -d "%INPUT_DIRECTORY%" -frate %FRAME_RATE% ^
  -p pipelines\detector_yolo_default.pipe ^
  -s input:video_filename=input_list.txt ^
  -s detector:detector:darknet:net_config=deep_training\yolo_v2.cfg ^
  -s detector:detector:darknet:weight_file=deep_training\models\yolo_v2.backup ^
  -s detector:detector:darknet:class_names=deep_training\yolo_v2.lbl ^
  -s detector:detector:darknet:scale=1.4 ^
  -s detector_writer:file_name=deep_detections.csv

pause
