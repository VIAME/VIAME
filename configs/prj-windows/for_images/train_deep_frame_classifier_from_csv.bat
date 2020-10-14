@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=training_data

REM Disable warnings
SET KWIMAGE_DISABLE_C_EXTENSIONS=1

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

viame_train_detector.exe ^
  -i "%INPUT_DIRECTORY%" ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_netharn_resnet.viame_csv.conf" ^
  --threshold 0.0

pause
