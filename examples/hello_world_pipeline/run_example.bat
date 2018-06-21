@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

CALL .\..\..\setup_viame.bat

REM Run Pipeline

pipeline_runner.exe -p hello_world_detector.pipe

pause
