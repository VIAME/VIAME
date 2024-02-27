@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT=videos
SET OUTPUT=output
SET FRAME_RATE=5
SET PIPELINE=pipelines\detector_fish_without_motion.pipe

REM Extra resource utilization options
SET TOTAL_GPU_COUNT=1
SET PIPES_PER_GPU=1

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" ^
  -i "%INPUT%" -o %OUTPUT% -frate %FRAME_RATE% ^
  -p %PIPELINE% --no-reset-prompt ^
  -gpus %TOTAL_GPU_COUNT% -pipes-per-gpu %PIPES_PER_GPU%

PAUSE
