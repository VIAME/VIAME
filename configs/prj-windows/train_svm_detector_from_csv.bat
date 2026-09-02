@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=training_data

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

viame.exe train ^
  -i "%INPUT_DIRECTORY%" ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_detector_svm_over_generic_detections.conf" ^
  --threshold 0.0

PAUSE
