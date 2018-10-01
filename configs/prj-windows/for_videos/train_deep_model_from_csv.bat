@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=C:\Program Files\VIAME

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run pipeline

viame_train_detector.exe ^
  -i training_data ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_yolo_704.viame_csv.conf" ^
  --threshold 0.0

pause
