@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Processing options
SET INPUT_DIRECTORY=training_data
SET INITIAL_MODEL=category_models\trained_classifier.zip

REM Disable warnings
SET KWIMAGE_DISABLE_C_EXTENSIONS=1

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Adjust log level
SET KWIVER_DEFAULT_LOG_LEVEL=info

IF EXIST "%INITIAL_MODEL%" (
  viame_train_detector.exe ^
    -i "%INPUT_DIRECTORY%" ^
    -c "%VIAME_INSTALL%\configs\pipelines\train_netharn_resnet.viame_csv.conf" ^
    -s "detector_trainer:netharn:seed_model=%INITIAL_MODEL%" ^
    --threshold 0.0
) ELSE (
  IF EXIST "deep_training" (
    viame_train_detector.exe ^
      -i "%INPUT_DIRECTORY%" ^
      -c "%VIAME_INSTALL%\configs\pipelines\train_netharn_resmet_nf.viame_csv.conf" ^
      --threshold 0.0
  ) ELSE (
    ECHO Initial seed model or in progress training folder does not exist, exiting
  )
)

pause
