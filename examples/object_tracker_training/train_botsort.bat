@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

REM Train BoT-SORT Re-ID model with camera motion compensation
viame.exe train ^
  -i training_data ^
  -tt botsort ^
  --threshold 0.0

pause
