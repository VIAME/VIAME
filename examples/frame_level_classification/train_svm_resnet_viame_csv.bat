@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Processing options
SET INPUT_DIRECTORY=training_data

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

viame_train_detector.exe ^
  -i "%INPUT_DIRECTORY%" ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_svm_full_frame_classifier.viame_csv.conf" ^
  --threshold 0.0

pause
