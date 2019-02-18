@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run pipeline

viame_train_detector.exe ^
  -i training_data_habcam ^
  -c "%VIAME_INSTALL%/configs/pipelines/train_yolo_704.habcam.conf" ^
  -p "%VIAME_INSTALL%/configs/pipelines/train_split_and_stereo_aug.pipe" ^
  --threshold 0.0

pause
