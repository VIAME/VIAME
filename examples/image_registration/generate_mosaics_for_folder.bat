@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Processing options
SET INPUT=insert_here
SET OUTPUT=output

REM Extra resource utilization options
SET TOTAL_GPU_COUNT=1
SET PIPES_PER_GPU=1

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" -i "%INPUT%" -o "%OUTPUT%" -p auto --mosaic -gpus %TOTAL_GPU_COUNT% -pipes-per-gpu %PIPES_PER_GPU% -install "%VIAME_INSTALL%"

pause
