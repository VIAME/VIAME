@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

REM Train RF-DETR detector on 16-bit imagery with automatic normalization
REM The --normalize-16bit flag enables percentile normalization for non-8-bit imagery
viame.exe train ^
  --input-list input_list_arctic_seal_16bit.txt ^
  --input-truth groundtruth_arctic_seal_16bit.csv ^
  --labels labels_arctic_seal_16bit.txt ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_detector_rf_detr_b_560.conf" ^
  --normalize-16bit ^
  --threshold 0.0

pause
