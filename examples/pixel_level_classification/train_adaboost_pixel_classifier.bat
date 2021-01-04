@echo off

REM Set path to input directory
SET INPUT_DIRECTORY=training_data

REM Setup VIAME Paths (no need to set if installed to registry or already set up)
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

REM Disable warnings
SET KWIMAGE_DISABLE_C_EXTENSIONS=1

REM Run pipeline
viame_train_detector.exe ^
  -i %INPUT_DIRECTORY% ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_adaboost_pixel_classifier.viame_csv.conf" ^
  --threshold 0.0

pause
