@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

set VIAME_INSTALL=C:\Program Files\VIAME

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

ffmpeg.exe -i [video_name].avi -r 10 frames2/%%06d.png

pause
