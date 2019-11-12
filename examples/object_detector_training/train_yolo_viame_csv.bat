@echo off

REM Input locations and types

SET INPUT_DIRECTORY=training_data

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

viame_train_detector.exe ^
  -i "%INPUT_DIRECTORY%" ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_yolo_704.viame_csv.conf" ^
  --threshold 0.0

pause

