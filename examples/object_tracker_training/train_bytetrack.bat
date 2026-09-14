@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET VIAME_LOG_LEVEL=info

REM Train ByteTrack tracker parameters from groundtruth tracks
viame.exe train ^
  -i training_data ^
  --tracker bytetrack ^
  --threshold 0.0

pause
