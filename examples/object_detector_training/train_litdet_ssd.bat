@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

REM Disable warnings
SET KWIMAGE_DISABLE_C_EXTENSIONS=1

REM Run pipeline
viame.exe train ^
  -i training_data_mouss ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_detector_litdet_ssd.conf" ^
  --threshold 0.0

pause
