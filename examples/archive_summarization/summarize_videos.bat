@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

python.exe "%VIAME_INSTALL%\configs\process_video.py" --init -d INPUT_DIRECTORY ^
  --detection-plots ^
  -plot-threshold 0.25 -frate 2 -plot-smooth 2 ^
  -p pipelines\index_default.pipe

pause
