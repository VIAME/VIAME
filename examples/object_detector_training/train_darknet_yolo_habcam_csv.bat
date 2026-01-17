@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

REM Run pipeline
viame.exe train ^
  -i training_data_habcam ^
  -c "%VIAME_INSTALL%/configs/pipelines/train_detector_darknet_yolo_704.habcam.conf" ^
  --threshold 0.0

pause
