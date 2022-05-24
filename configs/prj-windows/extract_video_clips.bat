@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT=videos
SET OUTPUT=video_clips
SET FRAME_RATE=5
SET MAX_DURATION=05:00.00

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" ^
  -i "%INPUT%" -o %OUTPUT% -frate %FRAME_RATE% ^
  -p "pipelines/transcode_default.pipe" ^
  -s "video_writer:maximum_length=%MAX_DURATION%"

pause
