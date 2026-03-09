@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

REM Train tracker using adaptive selection
viame.exe train ^
  -i training_data ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_tracker_adaptive.conf" ^
  --threshold 0.0

pause
