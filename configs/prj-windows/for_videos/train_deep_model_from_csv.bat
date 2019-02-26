@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=training_data

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

viame_train_detector.exe ^
  -i training_data ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_yolo_544.viame_csv.conf" ^
  --threshold 0.0

pause
