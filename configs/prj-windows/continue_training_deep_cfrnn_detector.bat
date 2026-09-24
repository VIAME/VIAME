@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=training_data

REM Seed model: the pack written by training, or an unpacked folder
IF EXIST "trained_model.zip" (
  SET SEED_MODEL=trained_model.zip
) ELSE IF EXIST "trained_model" (
  SET SEED_MODEL=trained_model
) ELSE (
  SET SEED_MODEL=category_models\trained_detector.zip
)

REM Disable warnings
SET KWIMAGE_DISABLE_C_EXTENSIONS=1

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

IF EXIST "%SEED_MODEL%" (
  viame.exe train ^
    -i "%INPUT_DIRECTORY%" ^
    -c "%VIAME_INSTALL%\configs\pipelines\train_detector_netharn_cfrnn.conf" ^
    --init-weights "%SEED_MODEL%" ^
    --threshold 0.0
) ELSE (
  IF EXIST "deep_training" (
    viame.exe train ^
      -i "%INPUT_DIRECTORY%" ^
      -c "%VIAME_INSTALL%\configs\pipelines\train_detector_netharn_cfrnn.conf" ^
      --continue ^
      --threshold 0.0
  ) ELSE (
    ECHO Initial seed model or in progress training folder does not exist, exiting
  )
)

PAUSE
