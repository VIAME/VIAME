@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=videos
SET FRAME_RATE=5
SET START_TIME=00:00:00.00
SET CLIP_DURATION=00:05:00.00
SET OUTPUT_DIR=images

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

ffmpeg.exe -i [video_name].avi -r %FRAME_RATE% -ss %START_TIME% -t %CLIP_DURATION% %OUTPUT_DIR%/frame%%06d.png

pause
