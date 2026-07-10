@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

REM Train OC-SORT tracker parameters from groundtruth tracks
viame.exe train ^
  -i training_data ^
  -tt ocsort ^
  --threshold 0.0

pause
