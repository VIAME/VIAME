@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=videos
SET CACHE_DIRECTORY=database
SET FRAME_RATE=5

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Set current directory for project folder pipe
SET VIAME_PROJECT_DIR=%~dp0

python.exe "%VIAME_INSTALL%\configs\launch_annotation_interface.py" -d "%INPUT_DIRECTORY%" -c "%CACHE_DIRECTORY%" -frate %FRAME_RATE%

