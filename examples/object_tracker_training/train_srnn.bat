@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

REM Train SRNN multi-stage tracking model
viame.exe train ^
  -i training_data ^
  -tt srnn ^
  --threshold 0.0

pause
