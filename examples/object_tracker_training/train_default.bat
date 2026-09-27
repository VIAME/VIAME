@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

REM Train ByteTrack, or a registration-based tracker when the data needs one
viame.exe train ^
  -i training_data ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_tracker_default.conf" ^
  --threshold 0.0

pause
