@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

set VIAME_INSTALL=C:\Program Files\VIAME
set FRAME_RATE=10
set OUTPUT_DIR=images

call "%VIAME_INSTALL%\setup_viame.bat"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Run Pipeline

ffmpeg.exe -i [video_name].avi -r %FRAME_RATE% %OUTPUT_DIR%/frame%%06d.png

pause
