@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=videos
SET FRAME_RATE=5

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" --init -d "%INPUT_DIRECTORY%" -frate %FRAME_RATE% -p pipelines\index_default.res.pipe --build-index --ball-tree -install "%VIAME_INSTALL%"

pause
