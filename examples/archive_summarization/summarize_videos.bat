@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

viame.exe run --init -d INPUT_DIRECTORY ^
  --detection-plots ^
  -plot-threshold 0.25 -frate 2 -plot-smooth 2 ^
  -p pipelines\index_generic.pipe

pause
