@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Processing options
SET INPUT_DIRECTORY=training_data_mouss
SET INITIAL_MODEL=category_models\trained_detector.zip

REM Disable warnings
SET KWIMAGE_DISABLE_C_EXTENSIONS=1

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

IF EXIST "%INITIAL_MODEL%" (
  viame.exe train ^
    -i "%INPUT_DIRECTORY%" ^
    -c "%VIAME_INSTALL%\configs\pipelines\train_detector_netharn_cfrnn.conf" ^
    -s "detector_trainer:ocv_windowed:trainer:netharn:seed_model=%INITIAL_MODEL%" ^
    --threshold 0.0
) ELSE (
  IF EXIST "deep_training" (
    viame.exe train ^
      -i "%INPUT_DIRECTORY%" ^
      -c "%VIAME_INSTALL%\configs\pipelines\train_detector_netharn_cfrnn_nf.conf" ^
      -s "detector_trainer:ocv_windowed:skip_format=true" ^
      --threshold 0.0
  ) ELSE (
    ECHO Initial seed model or in progress training folder does not exist, exiting
  )
)

pause
