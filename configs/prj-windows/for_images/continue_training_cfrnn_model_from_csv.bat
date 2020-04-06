@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=training_data
SET INITIAL_MODEL=category_models\trained_detector.zip

REM Disable warnings
SET KWIMAGE_DISABLE_C_EXTENSIONS=1

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

viame_train_detector.exe ^
  -i "%INPUT_DIRECTORY%" ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_netharn_cascade.viame_csv.conf" ^
  -s "detector_trainer:ocv_windowed:trainer:netharn:seed_model=%INITIAL_MODEL%" ^
  --threshold 0.0

pause
