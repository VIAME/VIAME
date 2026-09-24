@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT=videos
SET OUTPUT=output
SET FRAME_RATE=5

REM Extra resource utilization options
SET TOTAL_GPU_COUNT=1
SET PIPES_PER_GPU=1

REM Trained model: the pack written by the training scripts, or the legacy folder
IF EXIST "trained_model.zip" (
  SET TRAINED_MODEL=trained_model.zip\detector.pipe
) ELSE (
  SET TRAINED_MODEL=category_models\detector.pipe
)

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Set current directory for project folder pipe
SET VIAME_PROJECT_DIR=%~dp0

viame.exe run ^
  -i "%INPUT%" -o %OUTPUT% -frate %FRAME_RATE% ^
  -p "%TRAINED_MODEL%" --no-reset-prompt ^
  -gpus %TOTAL_GPU_COUNT% -pipes-per-gpu %PIPES_PER_GPU%

PAUSE
